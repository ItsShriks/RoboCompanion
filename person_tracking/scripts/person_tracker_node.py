#!/usr/bin/env python3
# Author - https://github.com/ItsShriks
import rospy
import moveit_commander
import sys
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, Pose, Quaternion, TransformStamped
from mas_execution_manager.scenario_state_base import ScenarioStateBase
from moveit_commander import PlanningSceneInterface
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from stop_confirmation import SpeechRecognitionService as sr


class PersonFollow(ScenarioStateBase):
    def __init__(self, save_sm_state=False, **kwargs):
        ScenarioStateBase.__init__(self, 'person_follow',
                                  save_sm_state=save_sm_state,
                                  outcomes=['succeeded', 'failed', 'failed_after_retrying'])
        
        self.sm_id = kwargs.get('sm_id', 'mdr_person_follow')
        self.timeout = kwargs.get('timeout', 120.)
        self.number_of_retries = kwargs.get('number_of_retries', 0)
        self.retry_count = 0
        self.head = moveit_commander.MoveGroupCommander("head")
        self.stop_confirmation = sr()

        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        
        
        # Configuration parameters (can be overridden through kwargs)
        self.FRAME_WIDTH = kwargs.get('frame_width', 640)
        self.CENTER_THRESHOLD = kwargs.get('center_threshold', 100)
        self.DEPTH_THRESHOLD_NEAR = kwargs.get('depth_threshold_near', 800)
        self.DEPTH_THRESHOLD_FAR = kwargs.get('depth_threshold_far', 2000)
        self.TARGET_DISTANCE = kwargs.get('target_distance', 1.0)
        self.MOVEMENT_THRESHOLD = kwargs.get('movement_threshold', 50)
        self.speed = kwargs.get('speed', 0.2)
        self.angular_speed = kwargs.get('angular_speed', 0.08)
        self.sleep_speed = kwargs.get('sleep_speed', 5.0)
        
        # State variables
        self.current_distance = None
        self.last_depth = None
        self.latest_depth_image = None
        self.bridge = CvBridge()
        self.in_motion = False
        self.target_reached = False
        self.person_detected = False
        
        self.current_position = None
        self.current_orientation = None
        self.current_velocity = None

        
        # Publishers
        self.lateral_pub = rospy.Publisher('person_tracking/lateral_movement', String, queue_size=1)
        self.distance_pub = rospy.Publisher('person_tracking/distance_movement', String, queue_size=1)
        self.velocity_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
        
        
        # Subscribers setup occurs during execution to avoid premature callbacks
        self.image_sub = None
        self.depth_sub = None 
        self.laser_sub = None
        
        rospy.loginfo("Person Follow State initialized")
    def is_stop_gesture(self, landmarks):
        tips = [self.landmarks.landmark[i] for i in [8, 12, 16, 20]]
        base = self.landmarks.landmark[0]
        return all([tip.y > base.y for tip in tips])
    
    def odom_callback(self, msg):
        self.current_position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        quaternion = (orientation_q.x, orientation_q.y, quaternion.z, quaternion.w)
        euler = tf_trans.euler_from_quaternion(quaternion)
        self.current_orientation = euler  # (roll, pitch, yaw)
        
        self.current_velocity = msg.twist.twist.linear

        rospy.loginfo(f"Odometry - Position: ({self.current_position.x}, {self.current_position.y}, {self.current_position.z})")
        rospy.loginfo(f"Odometry - Orientation (Yaw): {self.current_orientation[2]}")
        rospy.loginfo(f"Odometry - Velocity: ({self.current_velocity.x}, {self.current_velocity.y}, {self.current_velocity.z})")

    def get_odometry(self):
        """Returns the current odometry data as a dictionary."""
        return {
            "position": self.current_position,
            "orientation": self.current_orientation,
            "velocity": self.current_velocity
        }
    
    def move_base_vel(self, vx, vy, vw):
        twist = Twist()
        twist.linear.x = self.vx
        twist.linear.y = self.vy
        twist.angular.z = self.vw / 180.0 * math.pi  # 「度」から「ラジアン」に変換します
        base_vel_pub.publish(twist)  # 速度指令をパブリッシュします


    def move_head_tilt(self, v):
        self.head.set_joint_value_target("head_tilt_joint", v)
        self.head.go()
    
        def move_head_pan(self, v):
            self.head.set_joint_value_target("head_pan_joint", v)
            return self.head.go()
        
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

    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            results_hands = self.hands.process(rgb_frame)
            
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    if self.is_stop_gesture(hand_landmarks):##
                        confirmed = self.stop_confirmation.get_and_confirm_input(initial_prompt="Do you want me to stop?")##
                        self.say("Stop gesture detected, stopping the robot")
                        return 'succeeded'
                    else:
                        pass
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
                
                if x < center_x - self.CENTER_THRESHOLD:
                    movement_direction = "LEFT"
                    self.move_left(self.angular_speed)
                elif x > center_x + self.CENTER_THRESHOLD:
                    movement_direction = "RIGHT"
                    self.move_right(self.angular_speed)
                else:
                    self.move_straight(self.speed)
                
                self.lateral_pub.publish(movement_direction)
                
                
                if self.latest_depth_image is not None:
                    depth = self.get_depth_at_point(self.latest_depth_image, x, y)
                    if depth is not None:
                        z_movement, _ = self.determine_z_movement(depth)
                        self.distance_pub.publish(z_movement)
                        
                        # Save current distance for state machine
                        self.current_distance = depth
                        
                        if depth < self.TARGET_DISTANCE:
                            self.move_back(self.speed)
                        elif depth > self.TARGET_DISTANCE:
                            self.move_forward(self.speed)
                        else:
                            self.stop_movement()

                        # Check if target distance is reached (with a small tolerance)
                        if abs(depth - self.TARGET_DISTANCE) < 0.4:
                            rospy.loginfo("Target distance reached!")
                            self.say("Do you want me to still follow you, or should i pick up the bag ?")
                            self.target_reached = True
                
                filename = "./person_tracker.png"
                cv2.imwrite(filename, frame)
            else:
                self.person_detected = False
                self.lateral_pub.publish("NO_PERSON")
                self.distance_pub.publish("NO_PERSON")                
        except Exception as e:
            rospy.logerr(f"Error processing RGB frame: {str(e)}")

    def laser_scan_callback(self, msg):
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        
        min_range = min(ranges)
        min_range_index = ranges.index(min_range)
        min_angle = angle_min + min_range_index * angle_increment

        rospy.loginfo(f"Closest object at distance {min_range} meters, angle: {min_angle} radians")
        
        while True:
            if min_range < 0.3:
                rospy.logwarn("Object is too close! Stopping the robot.")
                self.say("Object is too close !")
                self.stop_movement()
                
                self.say("Obstacle detected, waiting for obstacle to pass")
                rospy.sleep(5.0)
                break  # exit the loop after handling the obstacle
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
    
    def move_straight(self, speed):
        twist = Twist()
        twist.linear.x = self.speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving Straight")
    
    def move_forward(self, speed):
        twist = Twist()
        twist.linear.x = self.speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving Forward")
    
    def move_back(self, speed):
        twist = Twist()
        twist.linear.x = -self.speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving Back")

    def stop_movement(self):
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        self.velocity_pub.publish(twist)
        rospy.loginfo("Stopping")

    def setup_subscribers(self):
        """Set up subscribers - called during execute to avoid premature callbacks"""
        self.image_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth/image_raw', Image, self.depth_callback)
        self.laser_sub = rospy.Subscriber('/hsrb/base_scan', LaserScan, self.laser_scan_callback)
        self.odom_sub = rospy.Subscriber('/hsrb/odom', Odometry, self.odom_callback)
        
    def clean_up_subscribers(self):
        """Clean up subscribers to avoid callback when state is not active"""
        if self.image_sub:
            self.image_sub.unregister()
        if self.depth_sub:
            self.depth_sub.unregister()
        if self.laser_sub:
            self.laser_sub.unregister()
            
    def execute(self, userdata):
        """Main state execution method called by the state machine"""
        # Reset state variables
        self.target_reached = False
        self.person_detected = False
        self.retry_count = 0
        self.last_depth = None
        self.person_location = None
        
        self.say('Following person')
        
        # Setup subscribers for callbacks
        self.setup_subscribers()
        
        # Main execution loop
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            # Check if the timeout has been exceeded
            if (rospy.Time.now() - start_time).to_sec() > self.timeout:
                self.say('Timeout reached while following person')
                self.clean_up_subscribers()
                self.stop_movement()
                
                if self.retry_count < self.number_of_retries:
                    self.retry_count += 1
                    rospy.loginfo(f"Retrying person follow (attempt {self.retry_count})")
                    return 'failed'
                else:
                    return 'failed_after_retrying'
            
            # Check if target distance has been reached
            if self.target_reached:
                self.say('Target distance reached')
                self.clean_up_subscribers()
                self.stop_movement()
                
                # Set output keys
                if self.person_location:
                    userdata.person_location = self.person_location
                
                return 'succeeded'
            
            
            
            # Check if person is not detected for too long
            if not self.person_detected and self.retry_count == 0:
                # Person not detected, but we'll give it some time
                
                self.say('Looking for person')
                
                rospy.sleep(10.0)
                self.move_head_tilt(0)
                rospy.sleep(2.0)
                
                if not self.person_detected:
                    self.retry_count += 1
                    rospy.loginfo(f"Person not detected, retrying ({self.retry_count})")
                    
                    if self.retry_count >= self.number_of_retries:
                        self.clean_up_subscribers()
                        self.stop_movement()
                        self.say('Person lost, aborting follow')
                        return 'failed_after_retrying'
            
            rate.sleep()
        
        # If we get here, ROS was shutdown
        self.clean_up_subscribers()
        self.stop_movement()
        return 'failed'
