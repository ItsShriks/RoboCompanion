import rospy
import smach
from geometry_msgs.msg import PoseStamped
from mas_execution_manager.scenario_state_base import ScenarioStateBase


class EstimatePose(ScenarioStateBase):
    """A mockup state for returning a predefined pose"""

    def __init__(self, **kwargs):
        #smach.State.__init__(self, "estimate_pose_mockup",
        #                     output_keys=["lever_pose"],
        #                     outcomes=["succeeded"])
        self.sm_id = kwargs.get("sm_id", "")
        self.state_name = kwargs.get("state_name", "estimate_pose_mockup")
        save_sm_state = kwargs.get("save_sm_state", False)
        ScenarioStateBase.__init__(self, self.state_name,
                                   save_sm_state=save_sm_state,
                                   output_keys=["lever_pose"],
                                   outcomes=['succeeded'])
        self.frame_id = kwargs.get("frame", "base_link")
        # expect position in [x, y. z] format
        self.position = kwargs.get("position", None)
        if not isinstance(self.position, list) or len(self.position) != 3:
            raise ValueError("position arg is not a list of length 3")
        # expect orientation in [x, y, z, w] format
        self.orientation = kwargs.get('orientation', None)
        if not isinstance(self.orientation, list) or len(self.orientation) != 4:
            raise ValueError("orientation arg is not a list of length 4")
        self.pose = PoseStamped()
        self.pose.header.frame_id = self.frame_id
        self.pose.pose.position.x = self.position[0]
        self.pose.pose.position.y = self.position[1]
        self.pose.pose.position.z = self.position[2]
        self.pose.pose.orientation.x = self.orientation[0]
        self.pose.pose.orientation.y = self.orientation[1]
        self.pose.pose.orientation.z = self.orientation[2]
        self.pose.pose.orientation.w = self.orientation[3]

    def execute(self, userdata):
        """Add a PoseStamped message to userdata"""
        rospy.loginfo(f"mockup pose estimation: frame={self.frame_id}, "
                      f"position={self.position}, orientation={self.orientation}")
        userdata.lever_pose = self.pose
        return "succeeded"

