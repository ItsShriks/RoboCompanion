#!/usr/bin/env python3
# Author - https://github.com/ItsShriks
import moveit_commander
import rospy
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from moveit_commander import PlanningSceneInterface


class HeadFollow:
    def __init__(self):
        self.head = moveit_commander.MoveGroupCommander("head")
        rospy.init_node('head_follow')
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()        
        
        self.FRAME_WIDTH = 640
        self.CENTER_THRESHOLD = 100
        self.DEPTH_THRESHOLD_NEAR = 800
        self.DEPTH_THRESHOLD_FAR = 2000
        self.TARGET_DISTANCE = 1.0
        self.MOVEMENT_THRESHOLD = 50  
        self.speed = 0.05
        self.angular_speed = 0.05
        self.sleep_speed = 5.0
        
        self.current_distance = None
        self.last_depth = None
        self.latest_depth_image = None
        self.bridge = CvBridge()
        self.in_motion = False
        self.is_centered = False  # Flag to track if person is centered
        self.last_position = None  # To track the last position of the person

        self.lateral_pub = rospy.Publisher('head_tracking/lateral_movement', String, queue_size=1)
        self.distance_pub = rospy.Publisher('head_tracking/distance_movement', String, queue_size=1)
        
        self.velocity_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
        
        self.image_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        
        self.laser_sub = rospy.Subscriber('/hsrb/base_scan', LaserScan, self.laser_scan_callback)
        
        rospy.loginfo("Person Follow Node initialized")

    
    def move_head_pan(self, v):
        self.head.set_joint_value_target("head_pan_joint", v)
        #self.head.go()
        return self.head.go()
    
    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # rospy.loginfo("Processing RGB")
                
            if results.pose_landmarks:  # Fixed indentation here
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
                
                center_x = w // 2
                movement_direction = "CENTER"
                speed = 0
                
                if not self.is_centered:
                    if x < center_x - self.CENTER_THRESHOLD:
                        movement_direction = "RIGHT"  # Invert direction here
                        self.move_right()  # Rotate to face the person
                    elif x > center_x + self.CENTER_THRESHOLD:
                        movement_direction = "LEFT"  # Invert direction here
                        self.move_left()  # Rotate to face the person    
                    else:
                        rospy.loginfo("Person is centered, no action required")
                else:
                    if x < center_x - self.CENTER_THRESHOLD or x > center_x + self.CENTER_THRESHOLD:
                        self.is_centered = False  # Reset the flag to allow movement again
                    
            else:  # Fixed indentation here as well
                self.lateral_pub.publish("NO_PERSON")
                self.distance_pub.publish("NO_PERSON")                
        except Exception as e:
            rospy.logerr(f"Error processing RGB frame: {str(e)}")

    def laser_scan_callback(self, msg):
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment

        #rospy.loginfo(f"Received laser scan with {len(ranges)} readings.")
        
        min_range = min(ranges)
        min_range_index = ranges.index(min_range)
        min_angle = angle_min + min_range_index * angle_increment

        #rospy.loginfo(f"Closest object at distance {min_range} meters, angle: {min_angle} radians")
        
        if min_range < 0.2:
            rospy.logwarn("Object is too close! Stopping the robot.")
            self.stop_movement()

    # def move_left(self):
    #     self.move_head_pan(-0.2)
    #     rospy.loginfo("Moving Left")
    
    # def move_right(self):
    #     self.move_head_pan(0.2)
    #     rospy.loginfo("Moving Right")
    
    # def stop_movement(self):
    #     self.move_head_pan(0)
    #     rospy.loginfo("Stopping")
    
    def move_left(self, speed):
        # Tilt the head to the left (you can adjust the tilt angle as needed)
        
        twist = Twist()
        twist.angular.z = self.angular_speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving Left")

    def move_right(self, speed):
        # Tilt the head to the right (you can adjust the tilt angle as needed)
        
        twist = Twist()
        twist.angular.z = -self.angular_speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving Right")
    
    def stop_movement(self):
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        self.velocity_pub.publish(twist)
        rospy.loginfo("Stopping")

    def shutdown_hook(self):
        rospy.loginfo("Shutting down Head Follow Node")
        
if __name__ == '__main__':
    try:
        tracker = HeadFollow()
        rospy.on_shutdown(tracker.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass