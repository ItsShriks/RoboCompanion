#!/usr/bin/env python3
# Author - https://github.com/ItsShriks

import rospy
import math
import mediapipe as mp
import cv2
import numpy as np
import moveit_commander
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from mas_execution_manager.scenario_state_base import ScenarioStateBase
from moveit_commander import PlanningSceneInterface
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class Greeting(ScenarioStateBase):
    def __init__(self, save_sm_state=False, **kwargs):
        ScenarioStateBase.__init__(self, 'greeting',
                                  save_sm_state=save_sm_state,
                                  outcomes=['succeeded', 'failed', 'failed_after_retrying'])
        
        self.sm_id = kwargs.get('sm_id', 'mdr_greeting')
        self.timeout = kwargs.get('timeout', 120.)
        self.number_of_retries = kwargs.get('number_of_retries', 0)
        self.retry_count = 0
        self.head = moveit_commander.MoveGroupCommander("head")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        #self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth/image_raw', Image, self.depth_callback)
        self.head = moveit_commander.MoveGroupCommander('head')
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        self.FRAME_WIDTH = kwargs.get('frame_width', 640)
        self.CENTER_THRESHOLD = kwargs.get('center_threshold', 100)
        self.DEPTH_THRESHOLD_NEAR = kwargs.get('depth_threshold_near', 800)
        self.DEPTH_THRESHOLD_FAR = kwargs.get('depth_threshold_far', 2000)                                                        
        self.TARGET_DISTANCE = kwargs.get('target_distance', 1.0)
        self.MOVEMENT_THRESHOLD = kwargs.get('movement_threshold', 50)
        self.speed = kwargs.get('speed', 0.2)
        self.angular_speed = kwargs.get('angular_speed', 0.08)
        self.sleep_speed = kwargs.get('sleep_speed', 5.0)
        self.person_detected = False
        
        self.latest_depth_image = None
        self.bridge = CvBridge()
        self.in_motion = False
        self.target_reached = False
        self.person_detected = False
        
   
    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
                
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
                
                # Save person location for state machine
                self.person_location = {'x': x, 'y': y, 'frame_width': w, 'frame_height': h}                                                                       
                center_x = w // 2
                movement_direction = "CENTER"
                # if x < center_x - self.CENTER_THRESHOLD:
                #     movement_direction = "LEFT"
                #     self.move_left(self.angular_speed)
                # elif x > center_x + self.CENTER_THRESHOLD:
                #     movement_direction = "RIGHT"
                #     self.move_right(self.angular_speed)
                # else:
                #     self.stop_movement()
                
                #self.lateral_pub.publish(movement_direction)
                if self.latest_depth_image is not None:
                    depth = self.get_depth_at_point(self.latest_depth_image, x, y)
                    if depth is not None:
                        z_movement, _ = self.determine_z_movement(depth)
                        #self.distance_pub.publish(z_movement)
                        
                        # Save current distance for state machine
                        self.current_distance = depth
                        
                        
                        # Check if target distance is reached (with a small tolerance)        
                filename = "./person_tracker.png"
                cv2.imwrite(filename, frame)
            else:
                self.person_detected = False
        except Exception as e:
            rospy.logerr(f"Error processing RGB frame: {str(e)}")
    def move_head_tilt(self, v):
        self.head.set_joint_value_target("head_tilt_joint", v)
        self.head.go()                                                                               
            rospy.logerr(f"Error processing RGB frame: {str(e)}")

    def move_head_tilt(self, v):
        self.head.set_joint_value_target("head_tilt_joint", v)
        self.head.go()

    def execute(self, userdata):
        self.target_reached = False
        self.person_detected = False
        self.retry_count = 0
        self.say('Hello, I am ready to Follow you')
        self.say('Please stand in front of me')
        #self.say('Detecting a person')
        #self.setup_subscribers()
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Check if the timeout has been exceeded
            if (rospy.Time.now() - start_time).to_sec() > self.timeout:
                self.say('Timeout reached while following person')
                if self.retry_count < self.number_of_retries:
                        self.retry_count += 1
                        rospy.loginfo(f"Retrying person follow (attempt {self.retry_count})")
                        return 'failed'
                else:
                    return 'failed_after_retrying'   
        if self.person_detected == True:
            self.say('Person Detected')
            self.move_head_tilt(-0.5)
            self.move_head_tilt(0.0)
            #self.clean_up_subscribers()
            #self.stop_movement()
            return 'succeeded'
        else:
            return 'failed'
