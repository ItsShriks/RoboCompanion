import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge

class BagPickup:
    def __init__(self):
        rospy.init_node('bag_pickup', anonymous=True)
        
        self.bridge = CvBridge()
        
        self.lateral_pub = rospy.Publisher('bag_pickup/lateral_movement', String, queue_size=1)
        self.distance_pub = rospy.Publisher('bag_pickup/distance_movement', String, queue_size=1)
        self.velocity_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
        self.image_pub = rospy.Publisher('/bag_pickup/image', Image, queue_size=1)  # New publisher

        self.image_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth/image_raw', Image, self.depth_callback)
        self.laser_sub = rospy.Subscriber('/hsrb/base_scan', LaserScan, self.laser_scan_callback)
        
        self.pose_model = YOLO("yolov8n-pose.pt")
        self.object_model = YOLO("yolov8n.pt")
        
    def depth_callback(self, msg):
        try:
            point_cloud = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            
            if not point_cloud:
                rospy.logwarn("Empty point cloud received")
                return
            
            depth_image = np.full((480, 640), np.nan, dtype=np.float32)

            for point in point_cloud:
                x, y, z = point
                pixel_x = int((x + 1) * (640 / 2))  
                pixel_y = int((y + 1) * (480 / 2))  
                if 0 <= pixel_x < 640 and 0 <= pixel_y < 480:
                    depth_image[pixel_y, pixel_x] = z

            self.latest_depth_image = depth_image
        except Exception as e:
            rospy.logerr(f"Error processing depth image from PointCloud2: {str(e)}")

    def get_depth_at_point(self, depth_image, x, y, window_size=5):
        if depth_image is None:
            return None
            
        height, width = depth_image.shape
        x_start = max(0, x - window_size)
        x_end = min(width, x + window_size)
        y_start = max(0, y - window_size)
        y_end = min(height, y + window_size)
        
        window = depth_image[y_start:y_end, x_start:x_end]
        valid_depths = window[window > 0]
        if len(valid_depths) > 0:
            return np.median(valid_depths)
        return None

    def laser_scan_callback(self, msg):
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment

        rospy.loginfo(f"Received laser scan with {len(ranges)} readings.")
        
        min_range = min(ranges)
        min_range_index = ranges.index(min_range)
        min_angle = angle_min + min_range_index * angle_increment

        #rospy.loginfo(f"Closest object at distance {min_range} meters, angle: {min_angle} radians")
        
        if min_range < 0.2:
            rospy.logwarn("Object is too close! Stopping the robot.")
            self.stop_movement()


    def rgb_callback(self, msg):
        frame = self.convert_ros_image_to_cv2(msg)
        height, width = frame.shape[:2]
        display_frame = frame.copy()
        if not hasattr(self, 'object_model'):
            rospy.logerr("object_model is not initialized!")
            return
        

        object_results = self.object_model(frame)
        rospy.loginfo("Object detection successful")
        bags = []
        if len(object_results) > 0 and hasattr(object_results[0], 'boxes') and hasattr(object_results[0].boxes, 'data'):
            for det in object_results[0].boxes.data:
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) == 26 and conf > 0.5:  # Class 26 is 'handbag' in COCO dataset
                        bags.append((int(x1), int(y1), int(x2), int(y2)))
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        pose_results = self.pose_model(frame)
        pointed_bags = set()
        for result in pose_results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                if hasattr(result.keypoints, 'data') and result.keypoints.data is not None:
                    try:
                        keypoints = result.keypoints.data.cpu().numpy()
                    except:
                        continue

                    for person_keypoints in keypoints:
                        if person_keypoints.shape[0] <= 10:
                            continue

                        for arm in [(7, 9), (8, 10)]:  # Left and Right arm keypoints
                            if np.all(person_keypoints[arm[0]][:2] > 0) and np.all(person_keypoints[arm[1]][:2] > 0):
                                elbow = person_keypoints[arm[0]][:2].astype(int)
                                wrist = person_keypoints[arm[1]][:2].astype(int)
                                intersection = self.extend_line_get_endpoint(wrist, elbow, width, height)
                                cv2.line(display_frame, tuple(wrist), intersection, (0, 0, 255), 2)
                                for i, (bx1, by1, bx2, by2) in enumerate(bags):
                                    if self.line_intersects_box(wrist, intersection, (bx1, by1, bx2, by2)):
                                        pointed_bags.add(i)

        for i in pointed_bags:
            if i < len(bags):
                bx1, by1, bx2, by2 = bags[i]
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                cv2.putText(display_frame, "POINTED", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.publish_image(display_frame)  # Publish processed image

    def publish_image(self, frame):
        """Convert OpenCV image to ROS Image message and publish it."""
        try:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.image_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"Failed to publish image: {e}")

    def extend_line_get_endpoint(self, point2, point1, width, height):
        x1, y1 = point1
        x2, y2 = point2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return (x2, y2)
        magnitude = np.sqrt(dx**2 + dy**2)
        dx /= magnitude
        dy /= magnitude
        t_values = []
        if dx < 0:
            t_values.append(-x2 / dx)
        if dx > 0:
            t_values.append((width-1 - x2) / dx)
        if dy < 0:
            t_values.append(-y2 / dy)
        if dy > 0:
            t_values.append((height-1 - y2) / dy)
        valid_t_values = [t for t in t_values if t >= 0]
        if valid_t_values:
            t = min(valid_t_values)
            x_end = int(x2 + dx * t)
            y_end = int(y2 + dy * t)
            return (x_end, y_end)
        else:
            return (x2, y2)

    def line_intersects_box(self, line_start, line_end, box):
        x1, y1, x2, y2 = box
        x1_line, y1_line = line_start
        x2_line, y2_line = line_end
        if (x1 <= x1_line <= x2 and y1 <= y1_line <= y2) or (x1 <= x2_line <= x2 and y1 <= y2_line <= y2):
            return True
        return False

    def convert_ros_image_to_cv2(self, ros_image):
        return np.frombuffer(ros_image.data, dtype=np.uint8).reshape((ros_image.height, ros_image.width, -1))

if __name__ == '__main__':
    node = BagPickup()
    rospy.spin()
