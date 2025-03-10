#!/usr/bin/env python3

import rospy
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from mas_execution_manager.scenario_state_base import ScenarioStateBase
import moveit_commander

class Greeting(ScenarioStateBase):
    def __init__(self, save_sm_state=False, **kwargs):
        ScenarioStateBase.__init__(self, 'greeting',
                                  save_sm_state=save_sm_state,
                                  outcomes=['succeeded', 'failed', 'failed_after_retrying'])
        
        self.sm_id = kwargs.get('sm_id', 'mdr_greeting')
        self.timeout = kwargs.get('timeout', 120.)
        self.number_of_retries = kwargs.get('number_of_retries', 0)
        self.retry_count = 0
        
        # Initialize MoveIt for head control
        self.head = moveit_commander.MoveGroupCommander("head")
        
        # Initialize CV bridge and mediapipe pose detection
        self.bridge = CvBridge()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5  # Slightly reduced to be more responsive
        )
        
        # Initialize parameters
        self.CENTER_THRESHOLD = kwargs.get('center_threshold', 100)
        self.person_detected = False
        self.greeting_done = False
        self.detection_start_time = None
        self.continuous_detection_time = 2.0  # Reduced to 2 seconds for faster response
        
        # Subscribe to the RGB image topic
        self.image_sub = rospy.Subscriber(
            '/hsrb/head_rgbd_sensor/rgb/image_raw', 
            Image, 
            self.rgb_callback,
            queue_size=1
        )
        
        # Publisher for status updates
        self.status_pub = rospy.Publisher('greeting_status', String, queue_size=1)
        
        rospy.loginfo("Greeting behavior initialized")

    def rgb_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Person detected
                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark
                
                # Create bounding box
                x_coords = [int(landmark.x * w) for landmark in landmarks]
                y_coords = [int(landmark.y * h) for landmark in landmarks]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Get nose position for tracking
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                x = int(nose.x * w)
                y = int(nose.y * h)
                
                # Check if person is centered enough
                center_x = w // 2
                is_centered = abs(x - center_x) < self.CENTER_THRESHOLD
                
                # Update detection status
                if not self.person_detected:
                    if self.detection_start_time is None:
                        self.detection_start_time = rospy.get_time()
                        rospy.loginfo("Person initially detected, starting confirmation timer")
                    elif rospy.get_time() - self.detection_start_time > self.continuous_detection_time:
                        self.person_detected = True
                        self.status_pub.publish("PERSON_CONFIRMED")
                        rospy.loginfo("Person confirmed after continuous detection")
                        return 'succeeded'
                
                # Display status on frame
                status_text = "CENTERED" if is_centered else "NOT_CENTERED"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save debug image
                # filename = "/tmp/person_greeting.png"
                # cv2.imwrite(filename, frame)
            else:
                # No person detected
                if self.detection_start_time is not None:
                    rospy.loginfo("Person detection lost")
                self.detection_start_time = None
                self.status_pub.publish("NO_PERSON")
                self.say('I still cannot see you. Please stand directly in front of me.')
                rospy.sleep(0.5)  # Short pause between sentences
                
        except Exception as e:
            rospy.logerr(f"Error in rgb_callback: {str(e)}")

    def move_head_to_neutral(self):
        """Position head to look forward at human height"""
        try:
            self.head.set_joint_value_target("head_pan_joint", 0)
            self.head.set_joint_value_target("head_tilt_joint", -0.3)  # Slightly tilted down
            self.head.go(wait=True)
        except Exception as e:
            rospy.logerr(f"Error moving head: {str(e)}")

    def execute(self, userdata):
        rospy.loginfo("Starting greeting behavior")
        
        # Reset state variables
        self.person_detected = False
        self.greeting_done = False
        self.detection_start_time = None
        
        # Move head to neutral position
        self.move_head_to_neutral()
        
        # Start with invitation if no person is detected
        self.say("Hello, please stand in front of me so I can see you.")
        
        # Wait for person detection with timeout
        start_time = rospy.get_time()
        detection_wait_rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < self.timeout:
            # Check if person is detected continuously
            if self.person_detected and not self.greeting_done:
                rospy.loginfo("Person detected and confirmed")
                
                # Greet the person
                self.say("Hello there! Nice to see you.")
                rospy.sleep(0.5)  # Short pause between sentences
                self.say("I'm ready to follow you now. Please walk slowly in front of me.")
                self.say("I will follow you until you show me your hand to stop.")
                
                self.greeting_done = True
                self.status_pub.publish("GREETING_COMPLETED")
                return 'succeeded'
                
            # If timeout is reached with no person
            elif (rospy.get_time() - start_time) > self.timeout / 2 and not self.person_detected:
                if self.retry_count < self.number_of_retries:
                    self.retry_count += 1
                    self.say("I still don't see anyone. Please stand directly in front of me.")
                    start_time = rospy.get_time()  # Reset timer for another attempt
                else:
                    rospy.logwarn("Failed to detect person after retries")
                    return 'failed_after_retrying'
                    
            detection_wait_rate.sleep()
            
        # If we exit the loop without success
        if not self.greeting_done:
            rospy.logwarn("Failed to detect person within timeout")
            return 'failed'

    def clean_up(self):
        """Clean up resources before shutting down"""
        try:
            if self.pose:
                self.pose.close()
            if hasattr(self, 'image_sub') and self.image_sub is not None:
                self.image_sub.unregister()
            rospy.loginfo("Greeting behavior cleaned up")
        except Exception as e:
            rospy.logerr(f"Error during cleanup: {str(e)}")