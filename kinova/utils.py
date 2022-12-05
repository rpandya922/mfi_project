from __future__ import division
from __future__ import print_function

import math
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose, PoseStamped, Point
import transforms3d.quaternions as quat
import transforms3d.affines as aff
from mfi_kinova.srv import iterative_ik
import time

def ComposePoseFromTransQuat(data_frame):
    # assert (len(data_frame.shape) == 1 and data_frame.shape[0] == 7)
    pose = Pose()
    pose.position.x = data_frame[0]
    pose.position.y = data_frame[1]
    pose.position.z = data_frame[2]
    pose.orientation.w = data_frame[3]
    pose.orientation.x = data_frame[4]
    pose.orientation.y = data_frame[5]
    pose.orientation.z = data_frame[6]
    return pose

def run_ik(pose, curr_joint_state):
    if curr_joint_state == []:
        return
    request = pose[:]
    request.extend(list(np.array(curr_joint_state)/180*math.pi))
    rospy.wait_for_service("ik_solver")
    try:
        ik = rospy.ServiceProxy("ik_solver", iterative_ik)
        resp = ik(*request)
        return [resp.j1, resp.j2, resp.j3, resp.j4, resp.j5, resp.j6, resp.j7]
    except rospy.ServiceException as e:
        print(e)

cartesian_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=1)
joint_pub = rospy.Publisher("/siemens_demo/joint_cmd", Float64MultiArray, queue_size=1)
def publish_pose(desired_pose, curr_joint_state=None, cmd_type="joint"):
    if cmd_type == "joint":
        assert curr_joint_state is not None
        q_des = run_ik(desired_pose, curr_joint_state)
        if q_des is not None:
            q_des = np.array(q_des)*180/math.pi # convert to degrees
            joint_msg = Float64MultiArray()
            joint_msg.data = list(np.asarray(q_des).flatten())
            joint_pub.publish(joint_msg)
    elif cmd_type == "cartesian":
        pose_msg = PoseStamped()
        pose_msg.pose = ComposePoseFromTransQuat(desired_pose)
        cartesian_pub.publish(pose_msg)

def publish_joint_state(joint_state):
    joint_msg = Float64MultiArray()
    joint_msg.data = joint_state
    joint_pub.publish(joint_msg)

kinova_grasp_pub = rospy.Publisher("/siemens_demo/gripper_cmd", Float64MultiArray, queue_size=1)
def close_gripper():
    g_cmd = Float64MultiArray()
    g_cmd.data = [0]
    kinova_grasp_pub.publish(g_cmd)
    # wait for gripper to close 
    # TODO: get access to gripper status 
    time.sleep(1)

def open_gripper():
    g_cmd = Float64MultiArray()
    g_cmd.data = [1]
    kinova_grasp_pub.publish(g_cmd)
    # wait for gripper to open
    # TODO: get access to gripper status 
    time.sleep(1)
