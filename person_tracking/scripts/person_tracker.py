import cv2
import mediapipe as mp
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class PersonFollow():
    def __init__(self):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        
        # Configuration parameters
        self.FRAME_WIDTH = 640
        self.CENTER_THRESHOLD = 100
        self.DEPTH_THRESHOLD_NEAR = 800
        self.DEPTH_THRESHOLD_FAR = 2000
        self.TARGET_DISTANCE = 1.0
        self.MOVEMENT_THRESHOLD = 50
        self.speed = 0.2
        self.angular_speed = 0.08
        self.sleep_speed = 5.0
        
        # State variables
        self.person_detected = False
        self.latest_depth_image = None
        self.bridge = CvBridge()
        
        # Initialize ROS node and subscribers
        rospy.init_node('person_follower', anonymous=True)
        self.rgb_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)

    def is_stop_gesture(self, landmarks):
        tips = [landmarks.landmark[i] for i in [8, 12, 16, 20]]
        base = landmarks.landmark[0]
        return all([tip.y > base.y for tip in tips])
    
    def rgb_callback(self, msg):
        try:
            # Convert ROS image to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            results_hands = self.hands.process(rgb_frame)
            
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    if self.is_stop_gesture(hand_landmarks):
                        print("Stop gesture detected, stopping the robot")
                        return 'succeeded'
            
            if results.pose_landmarks:
                self.person_detected = True
                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark
                
                x_coords = [int(landmark.x * w) for landmark in landmarks]
                y_coords = [int(landmark.y * h) for landmark in landmarks]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                x = int(nose.x * w)
                y = int(nose.y * h)
                
                # Save person location
                self.person_location = {'x': x, 'y': y, 'frame_width': w, 'frame_height': h}
                
                center_x = w // 2
                movement_direction = "CENTER"
                
                if x < center_x - self.CENTER_THRESHOLD:
                    movement_direction = "LEFT"
                    self.move_left(self.angular_speed)
                elif x > center_x + self.CENTER_THRESHOLD:
                    movement_direction = "RIGHT"
                    self.move_right(self.angular_speed)
                else:
                    self.move_straight(self.speed)
                
                if self.latest_depth_image is not None:
                    depth = self.get_depth_at_point(self.latest_depth_image, x, y)
                    if depth is not None:
                        z_movement, _ = self.determine_z_movement(depth)
                        
                        # Save current distance
                        self.current_distance = depth
                        
                        if depth < self.TARGET_DISTANCE:
                            self.move_back(self.speed)
                        elif depth > self.TARGET_DISTANCE:
                            self.move_forward(self.speed)
                        else:
                            self.stop_movement()

                filename = "./person_tracker.png"
                cv2.imwrite(filename, frame)
            else:
                self.person_detected = False
        except Exception as e:
            print(f"Error processing RGB frame: {str(e)}")

    def depth_callback(self, msg):
        try:
            # Convert ROS depth image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.latest_depth_image = depth_image
        except Exception as e:
            print(f"Error processing depth frame: {str(e)}")

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

    def determine_z_movement(self, current_depth):
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

    def move_left(self, speed):
        print("Moving Left")

    def move_right(self, speed):
        print("Moving Right")
    
    def move_straight(self, speed):
        print("Moving Straight")
    
    def move_forward(self, speed):
        print("Moving Forward")
    
    def move_back(self, speed):
        print("Moving Back")

    def stop_movement(self):
        print("Stopping")

if __name__ == '__main__':
    person_follower = PersonFollow()
    
    # Keep the ROS node running
    rospy.spin()