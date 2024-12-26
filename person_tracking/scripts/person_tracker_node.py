#!/usr/bin/env python3

import rospy
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

class PersonTracker:
    def __init__(self):
        rospy.init_node('person_tracker_node')
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        # Constants
        self.FRAME_WIDTH = 640
        self.CENTER_THRESHOLD = 100
        self.DEPTH_THRESHOLD_NEAR = 800  # in mm
        self.DEPTH_THRESHOLD_FAR = 2000  # in mm
        self.MOVEMENT_THRESHOLD = 50  # mm for depth movement detection
        
        # Initialize state variables
        self.last_depth = None
        self.latest_depth_image = None
        
        # Initialize the CvBridge
        self.bridge = CvBridge()
        
        # Create publishers for lateral and distance movements
        self.lateral_pub = rospy.Publisher('person_tracking/lateral_movement', String, queue_size=1)
        self.distance_pub = rospy.Publisher('person_tracking/distance_movement', String, queue_size=1)
        
        # Subscribe to the camera feeds
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
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
                
                # Draw circle on nose position
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                
                # Lateral movement detection
                center_x = self.FRAME_WIDTH // 2
                if x < center_x - self.CENTER_THRESHOLD:
                    lateral_movement = "LEFT"
                elif x > center_x + self.CENTER_THRESHOLD:
                    lateral_movement = "RIGHT"
                else:
                    lateral_movement = "CENTER"
                
                # Publish lateral movement
                self.lateral_pub.publish(lateral_movement)
                
                # Depth movement detection
                if self.latest_depth_image is not None:
                    depth = self.get_depth_at_point(self.latest_depth_image, x, y)
                    if depth is not None:
                        z_movement, _ = self.determine_z_movement(depth)
                        
                        # Publish distance movement
                        self.distance_pub.publish(z_movement)
                        
                        # Draw information on frame
                        cv2.putText(frame, f"Distance: {depth:.0f}mm", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Lateral: {lateral_movement}", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Z-Movement: {z_movement}", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # If no person detected, publish "NO_PERSON" for both movements
                self.lateral_pub.publish("NO_PERSON")
                self.distance_pub.publish("NO_PERSON")
            
            # Display the frames
            cv2.imshow("Person Tracking", frame)
            
            if self.latest_depth_image is not None:
                depth_colormap = cv2.normalize(self.latest_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
                cv2.imshow("Depth View", depth_colormap)
            
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing RGB frame: {str(e)}")

    def shutdown_hook(self):
        cv2.destroyAllWindows()
        rospy.loginfo("Shutting down Person Tracker Node")

if __name__ == '__main__':
    try:
        tracker = PersonTracker()
        rospy.on_shutdown(tracker.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass