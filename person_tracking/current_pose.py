#!/usr/bin/env python

import rospy
from mas_execution_manager.scenario_state_base import ScenarioStateBase

from geometry_msgs.msg import PoseStamped

# Global variable to store the current pose
current_pose = None
class DetectObject(ScenarioStateBase):
    def __init__(self, save_sm_state=False, **kwargs):
        ScenarioStateBase.__init__(self, 'detect_lever',
                                   save_sm_state=save_sm_state,
                                   outcomes=['succeeded', 'failed'],
                                   output_keys=['initial_pose'])
        self.sm_id = kwargs.get('sm_id', '')
        self.state_name = kwargs.get('state_name', 'detect_lever')
        self.number_of_retries = kwargs.get('number_of_retries', 0)
        self.debug = kwargs.get('debug', False)
        self.retry_count = 3
        self.timeout = 120.
        self.initial_pose = None
        rospy.Subscriber('/amcl_pose', PoseStamped, self.pose_callback)

    def execute(self):
        if self.initial_pose is None:
            return 'failed'
        return 'succeeded'

    def pose_callback(self, msg):
        """Callback function to store the robot's current pose."""
        self.initial_pose = msg
        userdata.initial_pose = self.initial_pose
        rospy.loginfo("Updated current pose: %s", current_pose)

    # def pose_listener(self):
    #     """Initializes the node and subscribes to the pose topic."""
    #     rospy.init_node('pose_saver', anonymous=True)
        
    #     # Subscribe to the pose topic (Change '/amcl_pose' if needed)
    #     rospy.Subscriber('/amcl_pose', PoseStamped, self.pose_callback)
        
    #     rospy.loginfo("Listening to /amcl_pose...")
    #     rospy.spin()  # Keep the node running

# if __name__ == '__main__':
#     try:
#         pose_listener()
#     except rospy.ROSInterruptException:
#         pass
