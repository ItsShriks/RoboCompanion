#!/usr/bin/env python3
# Author - https://github.com/ItsShriks

import rospy
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from ultralytics import YOLO
import time

class PersonFollow:
    def __init__(self):
        rospy.init_node('person_follow')
        
        self.tracked_person_id = None
        self.tracked_person_kpts = None
        self.tracked_person_timeout = rospy.Time.now()
        self.tracking_expiry_time = rospy.Duration(5)  # Stop tracking if lost for 5 seconds
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.model = YOLO("yolo11n-pose.pt")
        self.model.classes = [0]
        
        
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

        self.lateral_pub = rospy.Publisher('person_tracking/lateral_movement', String, queue_size=1)
        self.distance_pub = rospy.Publisher('person_tracking/distance_movement', String, queue_size=1)
        
        self.velocity_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
        self.yolo_pub = rospy.Publisher('/person_tracking/yolo', Image, queue_size=1)
        
        self.image_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth/image_raw', Image, self.depth_callback)  # Removed duplicate subscription
        self.laser_sub = rospy.Subscriber('/hsrb/base_scan', LaserScan, self.laser_scan_callback)
        
        rospy.loginfo("Person Follow Node initialized")

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
            
            yolo_results = self.model(rgb_frame)
            rospy.loginfo("Processing RGB")
            
            
            if yolo_results[0]:
                annotated_frame = yolo_results[0].plot()
                yolo_pub = self.bridge.cv2_to_imgmsg(annotated_frame, "rgb8")
                self.yolo_pub.publish(yolo_pub)

                print("Detected Person")
                
                kpts = yolo_results[0].keypoints.data
                
                if kpts is not None:
                    for i, person_kpts in enumerate(kpts):
                        x = int(person_kpts[0][0])
                        y = int(person_kpts[0][1])
                        
                        if self.tracked_person_id is None:
                            self.tracked_person_id = i
                            self.tracked_person_kpts = person_kpts
                            self.tracked_person_timeout = rospy.Time.now()
                            print(f"Tracking Person ID: {self.tracked_person_id}")
                        
                        if i == self.tracked_person_id:
                            self.tracked_person_kpts = person_kpts
                            self.tracked_person_timeout = rospy.Time.now()
                            rospy.loginfo(f"Updating Tracked Person: {self.tracked_person_id} location: {x}, {y}")
                            pass
                        
                        if self.tracked_person_id is not None and rospy.Time.now() - self.tracked_person_timeout > self.tracking_expiry_time:
                            rospy.logwarn("Lost the tracked person! Resetting tracking...")
                            self.tracked_person_id = None
                            self.tracked_person_kpts = None
                            
                        if self.tracked_person_id == 0:
                            x = int(self.tracked_person_kpts[0][0])  # Nose X
                            y = int(self.tracked_person_kpts[0][1])  # Nose Y
                            center_x = frame.shape[1] // 2

                        if x < center_x - self.CENTER_THRESHOLD:
                            self.move_left(self.angular_speed)
                            rospy.loginfo("Moving Left")
                        elif x > center_x + self.CENTER_THRESHOLD:
                            self.move_right(self.angular_speed)
                            rospy.loginfo("Moving Right")
                        else:
                            self.move_straight(self.speed)
                            rospy.loginfo("Moving Straight")
                        # elif:
                        #     self.stop_movement()
        except Exception as e:
            rospy.logerr(f"Error processing RGB frame: {str(e)}")
            
    def laser_scan_callback(self, msg):
        try:
            ranges = msg.ranges
            if len(ranges) == 0:
                rospy.logwarn("Empty laser scan received")
                return
            min_range = min(ranges)
            if min_range < 0.2:
                rospy.logwarn("Object is too close! Stopping the robot.")
                self.stop_movement()             
        except Exception as e:
            rospy.logerr(f"Error processing RGB frame: {str(e)}")

    def move_left(self, speed):
        twist = Twist()
        twist.angular.z = self.angular_speed
        self.velocity_pub.publish(twist)
        rospy.loginfo("Moving Left")
        # time.sleep(0.1)
        
    def move_right(self, speed):
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
        self.velocity_pub.publish(twist)
        rospy.loginfo("Stopping")

    def shutdown_hook(self):
        rospy.loginfo("Shutting down Person Follow Node")
        
if __name__ == '__main__':
    try:
        tracker = PersonFollow()
        rospy.on_shutdown(tracker.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
