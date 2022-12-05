from __future__ import print_function
from __future__ import division

import rospy
from std_msgs.msg import Float64MultiArray

from utils import publish_pose

curr_joint_state = []

def joint_cb(msg):
    global curr_joint_state
    try:
        curr_joint_state = list(msg.data)
    except:
        pass

if __name__ == "__main__":

    rospy.init_node("test_cartesian")

    r = rospy.Rate(10)

    desired_pose = [0.49, 0.083, -0.15, 0, 0.707, 0.707, 0]
    # desired_pose = [0.3, 0.1, 0.1, 0, 0.707, 0.707, 0]
    # equivalent joint state: [339.39, 24.35, 183.52, 231.28, 356.95, 333.01, 75.52]
    joint_sub = rospy.Subscriber("/kinova/current_joint_state", Float64MultiArray, joint_cb, queue_size=1)

    while not rospy.is_shutdown():

        publish_pose(desired_pose, curr_joint_state)

        r.sleep()
