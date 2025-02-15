#Author - Shrikar Nakhye
#!/usr/bin/env python3

import rospy
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class PersonTracker:
    def __init__(self):
        rospy.init_node('person_follow')
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        self.FRAME_WIDTH = 640
        self.CENTER_THRESHOLD = 100
        self.DEPTH_THRESHOLD_NEAR = 800  
        self.DEPTH_THRESHOLD_FAR = 2000  
        self.MOVEMENT_THRESHOLD = 50  
        self.speed = 0.02
        self.angular_speed = 0.5
        
        self.current_distance = None
        
        self.last_depth = None
        self.latest_depth_image = None
        self.bridge = CvBridge()
        
        # Publishers for lateral and distance movements
        self.lateral_pub = rospy.Publisher('person_tracking/lateral_movement', String, queue_size=1)
        self.distance_pub = rospy.Publisher('person_tracking/distance_movement', String, queue_size=1)
        
        self.velocity_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
        # Subscribe to the camera feeds
        self.image_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/rectified_points', Image, self.depth_callback)
        
        rospy.loginfo("Person Tracker Node initialized")

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {str(e)}")

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

    def move_left(self):
        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = -self.angular_speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving Left")
        twist.linear.x = 0
        twist.angular.z = 0
        self.velocity_pub.publish(twist)
    
    def move_right(self):
        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = self.angular_speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving Right")

        twist.linear.x = 0
        twist.angular.z = 0
        self.velocity_pub.publish(twist)
        
    def move_straight(self):
        twist = Twist()
        twist.linear.x = self.speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving straight")
        twist.linear.x = 0
        twist.angular.z = 0
        self.velocity_pub.publish(twist)
    
    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                
                x = int(nose.x * self.FRAME_WIDTH)
                y = int(nose.y * frame.shape[0])
                
                # Lateral movement detection
                center_x = self.FRAME_WIDTH // 2
                if x < center_x - self.CENTER_THRESHOLD:
                    lateral_movement = "LEFT"
                    self.move_left()
                elif x > center_x + self.CENTER_THRESHOLD:
                    lateral_movement = "RIGHT"
                    self.move_right()
                else:
                    lateral_movement = "CENTER"
                    self.move_straight()
                # Publish lateral movement
                self.lateral_pub.publish(lateral_movement)
                
                # Depth movement detection
                if self.latest_depth_image is not None:
                    depth = self.get_depth_at_point(self.latest_depth_image, x, y)
                    if depth is not None:
                        z_movement, _ = self.determine_z_movement(depth)
                        self.distance_pub.publish(z_movement)
            else:
                # If no person detected, publish "NO_PERSON" for both movements
                self.lateral_pub.publish("NO_PERSON")
                self.distance_pub.publish("NO_PERSON")
        
        except Exception as e:
            rospy.logerr(f"Error processing RGB frame: {str(e)}")

    def shutdown_hook(self):
        rospy.loginfo("Shutting down Person Tracker Node")

if __name__ == '__main__':
    try:
        tracker = PersonTracker()
        rospy.on_shutdown(tracker.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
