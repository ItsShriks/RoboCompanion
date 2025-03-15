import rospy
import smach
import smach_ros
from mdr_composite_behaviours.base_manuver import *
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point, Twist

class InitState(smach.State):
    def __init__(self):
        smach.State.__init__(self, 
                            outcomes=['succeeded', 'failed'])
        
    def execute(self, userdata):
        rospy.loginfo('Initializing Help Me Carry task')
        rospy.sleep(2.0)  # Wait for all nodes to start
        return 'succeeded'

class FollowPersonState(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                            outcomes=['reached_car', 'lost_person', 'failed'])
        
        self.cmd_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
        self.lateral_sub = rospy.Subscriber('person_tracking/lateral_movement', String, self.lateral_callback)
        self.distance_sub = rospy.Subscriber('person_tracking/distance_movement', String, self.distance_callback)
        
        self.current_lateral = "UNKNOWN"
        self.current_distance = 0.0
        self.person_lost = False
        
    def lateral_callback(self, msg):
        self.current_lateral = msg.data
        if msg.data == "LOST":
            self.person_lost = True
            
    def distance_callback(self, msg):
        try:
            self.current_distance = float(msg.data.split(":")[1].strip("mm"))
        except:
            pass
            
    def execute(self, userdata):
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            if self.person_lost:
                return 'lost_person'
                
            twist = Twist()
            if self.current_lateral == "LEFT":
                twist.angular.z = 0.3
            elif self.current_lateral == "RIGHT":
                twist.angular.z = -0.3
            elif self.current_lateral == "CENTER":
                twist.linear.x = 0.3
                
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
            
            # Check if we've reached the car (implement your logic here)
            if rospy.Time.now() - start_time > rospy.Duration(120.0):  # Timeout
                return 'failed'

class DetectBagState(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                            outcomes=['bag_detected', 'failed'])
        self.pointing_sub = rospy.Subscriber('person_tracking/pointing_detected', Bool, self.pointing_callback)
        self.pointing_detected = False
        
    def pointing_callback(self, msg):
        self.pointing_detected = msg.data
        
    def execute(self, userdata):
        # Implement bag detection logic here
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            if self.pointing_detected:
                return 'bag_detected'
            if rospy.Time.now() - start_time > rospy.Duration(30.0):
                return 'failed'
            rospy.sleep(0.1)

def main():
    rospy.init_node('help_me_carry_sm')
    
    sm = smach.StateMachine(outcomes=['succeeded', 'failed'])
    
    with sm:
        smach.StateMachine.add('INIT', InitState(),
                             transitions={'succeeded':'FOLLOW_PERSON',
                                        'failed':'failed'})
                                        
        smach.StateMachine.add('FOLLOW_PERSON', FollowPersonState(),
                             transitions={'reached_car':'DETECT_BAG',
                                        'lost_person':'FOLLOW_PERSON',
                                        'failed':'failed'})
                                        
        smach.StateMachine.add('DETECT_BAG', DetectBagState(),
                             transitions={'bag_detected':'succeeded',
                                        'failed':'failed'})
    
    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('help_me_carry_server', sm, '/SM_ROOT')
    sis.start()
    
    # Execute the state machine
    outcome = sm.execute()
    
    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()