from __future__ import division
from __future__ import print_function

import rospy
from geometry_msgs.msg import PoseStamped
import tf
import time

def robot_pose_transform(msg):
    global listener, robot_pose_pub
    try:
        output_pose = listener.transformPose("world", msg)
    except:
        return

    robot_pose_pub.publish(output_pose)


if __name__ == "__main__":
    rospy.init_node("robot_pose_transform")
    r = rospy.Rate(10)

    listener = tf.TransformListener()

    # transforming kinova's end effector position 
    robot_pose_sub = rospy.Subscriber("/kinova/pose_tool_in_base", PoseStamped, robot_pose_transform, queue_size=1)
    robot_pose_pub = rospy.Publisher("/kinova/pose_tool_in_world", PoseStamped, queue_size=1)

    time.sleep(1)
    while not rospy.is_shutdown():
        r.sleep()
