import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from mas_execution_manager.scenario_state_base import ScenarioStateBase
from std_msgs.msg import String

class BagPickup:
    def __init__(self):
        rospy.init_node("bag_pickup", anonymous=True)
        self.bridge = CvBridge()

        # Subscribe to RGB and Depth image topics
        self.image_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image_raw", Image, self.depth_callback)

        # Publisher for processed images
        self.image_pub = rospy.Publisher("/bag_picker", Image, queue_size=10)

        # YOLO models
        self.pose_model = YOLO("yolov8n-pose.pt")  # Pose detection
        self.object_model = YOLO("yolov8n.pt")  # Object detection

        # Depth image placeholder
        self.depth_image = None

        rospy.loginfo("BagPickup node started. Subscribed to camera topics.")

    def depth_callback(self, msg):
        """Store latest depth image."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logwarn(f"Failed to process depth image: {e}")

    def image_callback(self, msg):
        """Handle RGB images and process them for object and pose detection."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting ROS image: {e}")
            return

        processed_frame = self.process_frame(frame)
        
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
            self.image_pub.publish(processed_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing processed image: {e}")

    def process_frame(self, frame):
        """Processes the frame to detect bottles and pointing gestures."""
        height, width = frame.shape[:2]
        display_frame = frame.copy()

        # Detect objects (bottles)
        object_results = self.object_model(frame)
        bottles = []

        # Extract bottle detections
        if len(object_results) > 0 and hasattr(object_results[0], 'boxes') and hasattr(object_results[0].boxes, 'data'):
            for det in object_results[0].boxes.data:
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 39 and conf > 0.5:  # Class 39: Bottles
                        bottles.append([int(x1), int(y1), int(x2), int(y2), False])  # False -> Not pointed

        # Detect human poses
        pose_results = self.pose_model(frame)

        for result in pose_results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                if hasattr(result.keypoints, 'data') and result.keypoints.data is not None:
                    try:
                        keypoints = result.keypoints.data.cpu().numpy()
                    except:
                        continue  

                    if keypoints.size == 0:
                        continue

                    for person_keypoints in keypoints:
                        if person_keypoints.shape[0] <= 10:
                            continue  

                        # LEFT arm
                        if np.all(person_keypoints[7][:2] > 0) and np.all(person_keypoints[9][:2] > 0):
                            elbow = person_keypoints[7][:2].astype(int)
                            wrist = person_keypoints[9][:2].astype(int)
                            intersection = self.extend_line_get_endpoint(wrist, elbow, width, height)
                            cv2.line(display_frame, tuple(wrist), intersection, (0, 0, 255), 2)

                            for bottle in bottles:
                                if self.line_intersects_box(wrist, intersection, (bottle[0], bottle[1], bottle[2], bottle[3])):
                                    bottle[4] = True  # Mark as pointed

                        # RIGHT arm
                        if np.all(person_keypoints[8][:2] > 0) and np.all(person_keypoints[10][:2] > 0):
                            elbow = person_keypoints[8][:2].astype(int)
                            wrist = person_keypoints[10][:2].astype(int)
                            intersection = self.extend_line_get_endpoint(wrist, elbow, width, height)
                            cv2.line(display_frame, tuple(wrist), intersection, (0, 0, 255), 2)

                            for bottle in bottles:
                                if self.line_intersects_box(wrist, intersection, (bottle[0], bottle[1], bottle[2], bottle[3])):
                                    bottle[4] = True  # Mark as pointed

        # Draw bounding boxes
        for x1, y1, x2, y2, pointed in bottles:
            color = (0, 255, 0) if pointed else (255, 0, 0)  # Green if pointed, otherwise blue
            label = "POINTED" if pointed else "Bottle"

            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            depth = self.get_depth(center_x, center_y)
            rospy.loginfo(f"Bottle at ({center_x}, {center_y}) - Depth: {depth:.2f}m - {'Pointed' if pointed else 'Not Pointed'}")

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{label} ({depth:.2f}m)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return display_frame

    def get_depth(self, x, y):
        """Retrieve depth value at given image coordinates."""
        if self.depth_image is not None:
            try:
                depth_value = self.depth_image[y, x] * 0.001  # Convert mm to meters
                return round(depth_value, 2)
            except:
                return -1.0  # Invalid depth
        return -1.0  # Depth not available

    def extend_line_get_endpoint(self, point2, point1, width, height):
        """Extends a line from point1 through point2 until it reaches the image boundary."""
        x1, y1 = point1
        x2, y2 = point2
        dx, dy = x2 - x1, y2 - y1

        if dx == 0 and dy == 0:
            return (x2, y2)

        magnitude = np.sqrt(dx**2 + dy**2)
        dx /= magnitude
        dy /= magnitude

        t_values = [t for t in [
            -x2 / dx if dx < 0 else (width-1 - x2) / dx if dx > 0 else None,
            -y2 / dy if dy < 0 else (height-1 - y2) / dy if dy > 0 else None
        ] if t is not None]

        if t_values:
            t = min(t_values)
            return int(x2 + dx * t), int(y2 + dy * t)
        return x2, y2

    def line_intersects_box(self, line_start, line_end, box):
        """Determines if a line intersects with a bounding box."""
        x1, y1, x2, y2 = box
        return (x1 <= line_start[0] <= x2 and y1 <= line_start[1] <= y2) or \
               (x1 <= line_end[0] <= x2 and y1 <= line_end[1] <= y2)

if __name__ == "__main__":
    BagPickup()
    rospy.spin()
