#!/usr/bin/env python3

import rospy
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from mas_execution_manager.scenario_state_base import ScenarioStateBase

class Greeting(ScenarioStateBase):
    def __init__(self, save_sm_state=False, **kwargs):
        ScenarioStateBase.__init__(self, 'greeting',
                                  save_sm_state=save_sm_state,
                                  outcomes=['succeeded', 'failed', 'failed_after_retrying'])
        
        self.sm_id = kwargs.get('sm_id', 'mdr_greeting')
        self.timeout = kwargs.get('timeout', 120.)
        self.number_of_retries = kwargs.get('number_of_retries', 0)
        self.retry_count = 0
        
        self.bridge = CvBridge()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.CENTER_THRESHOLD = kwargs.get('center_threshold', 100)
        self.person_detected = False
        self.greeting_done = False
        self.detection_start_time = None
        self.continuous_detection_time = 2.0
        
        self.image_sub = rospy.Subscriber(
            '/hsrb/head_rgbd_sensor/rgb/image_raw', 
            Image, 
            self.rgb_callback,
            queue_size=1
        )
        
        self.status_pub = rospy.Publisher('greeting_status', String, queue_size=1)
    
    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                self.status_pub.publish("PERSON_DETECTED")
                rospy.loginfo("Person detected")
                person_detected = True
                return
            elif self.detection_start_time is None:
                return
            else:
                if self.detection_start_time is not None:
                    rospy.loginfo("Person detection lost")
                self.detection_start_time = None
                self.status_pub.publish("NO_PERSON")
                
                rospy.sleep(0.5)
                    
        except Exception as e:
            rospy.logerr(f"Error in rgb_callback: {str(e)}")

    def move_head_to_neutral(self):
        try:
            self.head.set_joint_value_target("head_pan_joint", 0)
            self.head.set_joint_value_target("head_tilt_joint", -0.3)
            self.head.go(wait=True)
        except Exception as e:
            rospy.logerr(f"Error moving head: {str(e)}")

    def execute(self, userdata):
        rospy.loginfo("Starting greeting behavior")
        
        self.person_detected = False
        self.greeting_done = False
        self.detection_start_time = None
        
        self.move_head_to_neutral()
        
        self.say("Hello, please stand in front of me so I can see you.")
        if person_detected:
            self.say("Hello Operator, I see you.")
            return 'succeeded'
        else:
            self.say("I still cannot see you. Please stand directly in front of me.")
            rospy.sleep(10.0)
        
        start_time = rospy.get_time()
        detection_wait_rate = rospy.Rate(10)