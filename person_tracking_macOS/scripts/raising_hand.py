import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2

from ultralytics import YOLO
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge

class HandRaiseDetector:
    def __init__(self):
        self.pose_model = YOLO("yolov8n-pose.pt", verbose=False)
        self.bridge = CvBridge()
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.last_depth = None
        self.MOVEMENT_THRESHOLD = 0.05  # Tune based on expected movement range

        rospy.init_node("hand_raise_detector", anonymous=True)

        # ROS Subscribers
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.point_cloud_callback)
        
        self.image_pub = rospy.Publisher("/raised_hand_image", Image, queue_size=10)


        rospy.loginfo("HandRaiseDetector Node Initialized")
        self.run()

    def rgb_callback(self, msg):
        """Converts ROS Image message to OpenCV format for RGB images."""
        try:
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error processing RGB image: {str(e)}")

    def depth_callback(self, msg):
        """Converts ROS Image message to OpenCV format for Depth images."""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            rospy.logerr(f"Error processing Depth image: {str(e)}")
    def point_cloud_callback(self, msg):
        try:
            pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            self.latest_point_cloud = np.array(list(pc_data))
        except Exception as e:
            rospy.logerr(f"Error processing PointCloud2 data: {str(e)}")

    def get_depth_at_point(self, depth_image, x, y, window_size=5):
        """Gets the median depth value in a small window around the given pixel coordinates."""
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

    def get_3d_depth_at_point(self, x, y):
        """Get 3D depth (x, y, z) from PointCloud2 data."""
        if self.latest_point_cloud is not None:
            height, width = self.latest_rgb_image.shape[:2]
            point_index = y * width + x  # Calculate index of the pixel in the point cloud data
            if point_index < len(self.latest_point_cloud):
                point = self.latest_point_cloud[point_index]
                return point  # Return the (x, y, z) tuple
        return None

    def determine_z_movement(self, current_depth):
        """Determines if the detected person is moving closer, away, or staying still."""
        if self.last_depth is None:
            self.last_depth = current_depth
            return "STOPPED", 0
        
        depth_diff = current_depth - self.last_depth
        self.last_depth = current_depth
        
        if abs(depth_diff) < self.MOVEMENT_THRESHOLD:
            return "STOPPED", depth_diff
        elif depth_diff > 0:
            return "MOVING_AWAY", depth_diff
        else:
            return "MOVING_CLOSER", depth_diff

    def calculate_bounding_box(self, keypoints):
        """Calculates the bounding box of a person from pose keypoints."""
        x_min = y_min = float('inf')
        x_max = y_max = -float('inf')
        
        for keypoint in keypoints:
            x, y = keypoint[:2]
            if x > 0 and y > 0:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        return int(x_min), int(y_min), int(x_max), int(y_max)
    def is_new_bounding_box(self, x_min, y_min, x_max, y_max):
        """Check if the bounding box is new by comparing with previously detected boxes."""
        for (prev_x_min, prev_y_min, prev_x_max, prev_y_max) in self.detected_bboxes:
            # Check for overlap by using a simple threshold
            if not (x_max < prev_x_min or x_min > prev_x_max or y_max < prev_y_min or y_min > prev_y_max):
                return False  # If there's overlap, return False (it's not a new bounding box)
        return True  # No overlap, return True (it's a new bounding box)

    def run(self):
        """Runs the hand-raising detection loop using ROS image topics."""
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            if self.latest_rgb_image is None:
                rospy.logwarn("Waiting for RGB image...")
                rate.sleep()
                continue

            display_frame = self.latest_rgb_image.copy()
            pose_results = self.pose_model(self.latest_rgb_image, verbose=False)
            
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
                            if person_keypoints.shape[0] <= 16:
                                continue
                            
                            if np.all(person_keypoints[0][:2] > 0):
                                nose = person_keypoints[0][:2].astype(int)
                                hand_raised = False
                                
                                if np.all(person_keypoints[10][:2] > 0):
                                    left_wrist = person_keypoints[10][:2].astype(int)
                                    if left_wrist[1] < nose[1]:
                                        hand_raised = True
                                
                                if np.all(person_keypoints[9][:2] > 0):
                                    right_wrist = person_keypoints[9][:2].astype(int)
                                    if right_wrist[1] < nose[1]:
                                        hand_raised = True
                                
                                depth_value = self.get_depth_at_point(self.latest_depth_image, nose[0], nose[1])
                                movement_status, depth_diff = self.determine_z_movement(depth_value) if depth_value else ("UNKNOWN", 0)
                                
                                if hand_raised:
                                    x_min, y_min, x_max, y_max = self.calculate_bounding_box(person_keypoints)
                                    if self.is_new_bounding_box(x_min, y_min, x_max, y_max):
                                        # Draw bounding box
                                        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                        self.detected_bboxes.append((x_min, y_min, x_max, y_max))
                                    
                                    print(f"Bounding Box for Hand-Raising Person: (x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max})")
                                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                                    triangle_top = (nose[0], nose[1] - 50)
                                    triangle_left = (nose[0] - 15, nose[1] - 20)
                                    triangle_right = (nose[0] + 15, nose[1] - 20)
                                    
                                    triangle_pts = np.array([triangle_top, triangle_left, triangle_right], np.int32)
                                    triangle_pts = triangle_pts.reshape((-1, 1, 2))
                                    cv2.fillPoly(display_frame, [triangle_pts], (0, 255, 0))
                                    
                                    cv2.putText(display_frame, "HAND RAISED", 
                                               (nose[0] - 50, nose[1] - 60),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    rospy.loginfo(f"Person detected with raised hand: Nose {tuple(nose)}, Depth: {depth_value}, Movement: {movement_status}")
            try:
                ros_image = self.bridge.cv2_to_imgmsg(display_frame, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                self.image_pub.publish(ros_image)
            except Exception as e:
                rospy.logerr(f"Error publishing image: {str(e)}")
                
            cv2.imshow('Hand Raising Detection', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    HandRaiseDetector()