#! /usr/bin/env python
import rospy
import time
import rosplan_dispatch_msgs.msg as plan_dispatch_msgs
import diagnostic_msgs.msg as diag_msgs
from mas_execution_manager.scenario_state_base import ScenarioStateBase
import moveit_commander
import geometry_msgs.msg
from tf.transformations import quaternion_from_euler
import sys
import ast
import rospy
import roslib
import actionlib
from geometry_msgs.msg import PoseStamped
from mdr_move_arm_action.msg import MoveArmAction, MoveArmGoal
from mdr_move_base_action.msg import MoveBaseAction, MoveBaseGoal
from mas_tools.ros_utils import get_package_path


class ArmtoBowl(ScenarioStateBase):
    def __init__(self, save_sm_state=False,
                 move_arm_server='move_arm_server',
                 move_base_server='move_base_server',
                 base_elbow_offset=0.078,
                 gripper_offset=0.05,
                 **kwargs):
        ScenarioStateBase.__init__(self, 'goto_pose',
                                   save_sm_state=save_sm_state,
                                   outcomes=['succeeded', 'failed'],
                                   input_keys=['lever_pose'])
        self.sm_id = kwargs.get('sm_id', '')
        self.state_name = kwargs.get('state_name', 'goto_pose')
        self.number_of_retries = kwargs.get('number_of_retries', 0)
        self.debug = kwargs.get('debug', False)
        self.retry_count = 0
        self.timeout = 120.
        self.move_arm_server= move_arm_server
        self.move_base_server = move_base_server
        self.base_elbow_offset = base_elbow_offset
        self.gripper_offset = gripper_offset
        self.lever_pose = kwargs.get('lever_pose', None)
        self.gripper = moveit_commander.MoveGroupCommander("gripper", wait_for_servers=0.0)
        #self.reference_frame = "odom"
        self.goal = MoveArmGoal()
        self.goal.goal_type = MoveArmGoal.END_EFFECTOR_POSE
        self.goal.dmp_name = get_package_path('mdr_pickup_action', 'config',
                                              'trajectory_weights', 'weights_table_grasp.yaml')
        self.goal.dmp_tau=30.
        rospy.loginfo("receiving lever pose from state machine: ")
        rospy.loginfo(self.lever_pose)
        

        try:
            self.client = actionlib.SimpleActionClient(self.move_arm_server, MoveArmAction)
            rospy.loginfo('[move_arm_to_pose] Waiting for %s server', self.move_arm_server)
            self.client.wait_for_server()
        except Exception as exc:
            rospy.logerr('[move_arm_to_pose] %s', str(exc))

        try:
            self.move_base_client = actionlib.SimpleActionClient(self.move_base_server, MoveBaseAction)
            rospy.loginfo('[pickup] Waiting for %s server', self.move_base_server)
            self.move_base_client.wait_for_server()
        except Exception as exc:
            rospy.logerr('[pickup] %s server does not seem to respond: %s',
                         self.move_base_server, str(exc))
        
        #self.whole_body.set_pose_reference_frame(self.reference_frame)
 

    def execute(self, userdata):
        rospy.loginfo('[move_arm_to_pose] starting move arm')
        self.lever_pose = userdata.lever_pose
        try:
            self.__align_base_with_pose(self.lever_pose)
            rospy.loginfo('[move_arm_to_pose] move arm to neutral')
            self.move_arm_to_neutral()
            # pose = PoseStamped()
            # pose.header.frame_id = 'base_link'

            # pose.pose.position.x = self.lever_pose[0]
            # pose.pose.position.y = self.lever_pose[1]
            # pose.pose.position.z = self.lever_pose[2]

            # pose.pose.orientation.x = self.lever_pose[3]
            # pose.pose.orientation.y = self.lever_pose[4]
            # pose.pose.orientation.z = self.lever_pose[5]
            # pose.pose.orientation.w = self.lever_pose[6]
            # self.goal.end_effector_pose = pose

            #self.goal.end_effector_pose.header.stamp = rospy.Time.now()
            self.goal.end_effector_pose.header.frame_id = self.lever_pose.header.frame_id
            self.goal.end_effector_pose.pose.position = self.lever_pose.pose.position
            self.goal.end_effector_pose.pose.position.y = self.base_elbow_offset
            self.goal.end_effector_pose.pose.position.z = 0.95
            self.goal.end_effector_pose.pose.position.x -= self.gripper_offset
            # self.goal.end_effector_pose.pose.position.x = pose.pose.position.x - 0.1
            self.goal.end_effector_pose.pose.orientation.x = 0
            self.goal.end_effector_pose.pose.orientation.y = 0
            self.goal.end_effector_pose.pose.orientation.z = 0
            self.goal.end_effector_pose.pose.orientation.w = 1
            rospy.logwarn("move arm pose")
            rospy.loginfo(self.goal.end_effector_pose)
            rospy.logwarn(f"goal type: {self.goal.goal_type}, tau={self.goal.dmp_tau}, dmp name: {self.goal.dmp_name}")
            timeout = 15.0
            rospy.loginfo('Sending action lib goal to move_arm_server')
            self.client.send_goal(self.goal)
            self.client.wait_for_result(rospy.Duration.from_sec(int(timeout)))
            rospy.logwarn("move arm result: ", self.client.get_result())

            ## gripper logic here======
            self.control_gripper(0.0)
            rospy.sleep(1)
            
            return 'succeeded'
        except Exception as e:
            rospy.logerr(e)
            return 'succeeded'

    def __align_base_with_pose(self, pose_base_link):
        '''Moves the base so that the elbow is aligned with the goal pose.

        Keyword arguments:
        pose_base_link -- a 'geometry_msgs/PoseStamped' message representing
                          the goal pose in the base link frame

        '''
        aligned_base_pose = PoseStamped()
        aligned_base_pose.header.frame_id = 'base_link'
        aligned_base_pose.header.stamp = rospy.Time.now()
        aligned_base_pose.pose.position.x = 0.
        aligned_base_pose.pose.position.y = pose_base_link.pose.position.y - self.base_elbow_offset
        aligned_base_pose.pose.position.z = 0.
        aligned_base_pose.pose.orientation.x = 0.
        aligned_base_pose.pose.orientation.y = 0.
        aligned_base_pose.pose.orientation.z = 0.
        aligned_base_pose.pose.orientation.w = 1.

        move_base_goal = MoveBaseGoal()
        move_base_goal.goal_type = MoveBaseGoal.POSE
        move_base_goal.pose = aligned_base_pose
        self.move_base_client.send_goal(move_base_goal)
        self.move_base_client.wait_for_result()

    def move_arm_to_neutral(self):
        move_arm_goal = MoveArmGoal()
        move_arm_goal.goal_type = MoveArmGoal.NAMED_TARGET
        move_arm_goal.named_target = "neutral"
        self.client.send_goal(move_arm_goal)
        self.client.wait_for_result()
        self.client.get_result()
        rospy.loginfo("Back to neutral position")
        rospy.sleep(5)

    def control_gripper(self, val):
        self.gripper.set_joint_value_target("hand_motor_joint", val)
        self.gripper.go()    
    
