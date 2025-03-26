import sys
import rospy
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from ultralytics import YOLO
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from cv_bridge import CvBridge
import moveit_commander
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped, Pose, Quaternion, TransformStamped
from mas_execution_manager.scenario_state_base import ScenarioStateBase
from moveit_commander import PlanningSceneInterface
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import person_tracker_node

# Add custom utilities directory
sys.path.append("/Users/shrikar/Library/Mobile Documents/com~apple~CloudDocs/Sem III/b-it bots @Home/RoboCompanion/person_tracking/refer_scripts")
import utils

class HeadToPose(ScenarioStateBase):
    def __init__(self, save_sm_state=False, **kwargs):
        ScenarioStateBase.__init__(self, 'head_to_pose',
                                  save_sm_state=save_sm_state,
                                  outcomes=['succeeded', 'failed', 'failed_after_retrying'])
        
        self.sm_id = kwargs.get('sm_id', 'mdr_head_to_pose')
        self.timeout = kwargs.get('timeout', 120.)
        self.number_of_retries = kwargs.get('number_of_retries', 0)
        self.retry_count = 0
        self.head = moveit_commander.MoveGroupCommander("head")
        
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.point_cloud_callback)
        
        rospy.loginfo("HeadToPose State Machine Initialized")
    
    def rgb_callback(self, msg):
        try:
            cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("RGB Image", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("RGB Callback Error: {}".format(e))
    
    def depth_callback(self, msg):
        try:
            depth_image = CvBridge().imgmsg_to_cv2(msg, "16UC1")
            depth_image = np.array(depth_image, dtype=np.uint16)
            cv2.imshow("Depth Image", depth_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Depth Callback Error: {}".format(e))
    
    def point_cloud_callback(self, msg):
        rospy.loginfo("Received PointCloud2 Data")
    
if __name__ == '__main__':
    try:
        node = HeadToPose()
        rospy.spin()
        print(person_tracker_node.bounding_box)
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()