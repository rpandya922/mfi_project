from __future__ import division
from __future__ import print_function

import math
from collections import deque
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose, Point
import transforms3d.quaternions as quat
import transforms3d.affines as aff
from mfi_kinova.srv import iterative_ik
import tf

robot_pts = deque()
buffer_size =10
marker_xyz = []
robot_joint_state = []

def run_ik(q):
    while robot_joint_state == []:
        pass
    request = q[:]
    request.extend(list(np.array(robot_joint_state)/180*math.pi))
    rospy.wait_for_service("ik_solver")
    try:
        ik = rospy.ServiceProxy("ik_solver", iterative_ik)
        resp = ik(*request)
        return [resp.j1, resp.j2, resp.j3, resp.j4, resp.j5, resp.j6, resp.j7]
    except rospy.ServiceException as e:
        print(e)

def robot_callback(msg):
    global robot_pts, buffer_size
    try:
        ee_pt = np.array([msg.x, msg.y, msg.z])
        robot_pts.append(ee_pt)
        while len(robot_pts) > buffer_size:
            robot_pts.popleft()
    except Exception:
        pass

def joint_callback(msg):
    global robot_joint_state
    # rospy.loginfo_throttle(1, msg.data)
    robot_joint_state = list(msg.data)
    # robot_joint_state[0] *= -1 # first joint angle is inverse to the kinematics chain

def ar_callback(msg):
    # messages on this topic are computed with kinova_base as the reference frame, so no need to transform them
    global marker_xyz
    try:
        marker = msg.data[0]
        if marker == 5:
            marker_xyz = msg.data[1:]
    except Exception:
        pass

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

def ComposeAffine(tx, ty, tz, w, rx, ry, rz, Z=np.ones(3)):
    R = quat.quat2mat([w, rx, ry, rz])
    T = [tx, ty, tz]
    mat = aff.compose(T, R, Z)
    return mat

if __name__ == "__main__":

    # initialize ros
    rospy.init_node("mfi_demo")
    r = rospy.Rate(10)
    kinova_joint_pub = rospy.Publisher("/siemens_demo/joint_cmd", Float64MultiArray, queue_size=1)
    kinova_grasp_pub = rospy.Publisher("/siemens_demo/gripper_cmd", Float64MultiArray, queue_size=1)
    kinova_gripper_sub = rospy.Subscriber("/kinova/current_position", Point, robot_callback, queue_size=1)
    kinova_joint_sub = rospy.Subscriber("/kinova/current_joint_state", Float64MultiArray, joint_callback, queue_size=1)
    ar_tag_sub = rospy.Subscriber("ar_marker_wrist_status", Float64MultiArray, ar_callback, queue_size=1)
    # listener = tf.TransformListener()

    # quaternion should be [w, x, y, z]
    desired_pose = [0.2, -0.4, -0.1, 0, 0.707, 0.707, 0] # TOOL_POS from siemens demo
    pose_reached = False

    # visual servoing with ar tag
    desired_xy = np.array([0.01, 0.05])
    Kp = 1.4*np.eye(2)
    marker = 5
    state = "align"
    sent_command = False

    while not rospy.is_shutdown():
        if state == "align":
            try:
                # (trans, rot) = listener.lookupTransform("/camera_color_frame", "/ar_marker_" + str(marker), rospy.Time(0))

                # xy = np.array(trans[:2])
                xy = marker_xyz[:2]
                err = xy - desired_xy
                # +y in gripper -> +x in pose of marker
                # +x in gripper -> +y in pose of marker
                # translate this error to a new pose command (coordinate system is different, flip x & y)
                gripper_err = np.array([-err[1], -err[0]])
                print("marker xy: " + str(xy))
                print()
                curr_pose = np.hstack((robot_pts[-1], [0, 0.707, 0.707, 0]))
                curr_pose[:2] = (Kp.dot(gripper_err)) + curr_pose[:2]

                # import pdb; pdb.set_trace()

                # TODO: make helper file that contains functions for publishing poses (either joint or pose commands)
                q_des = run_ik(list(curr_pose))
                if q_des is not None:
                    q_des = np.array(q_des)*180/math.pi # convert to degrees
                    # print(q_des)
                    joint_msg = Float64MultiArray()
                    joint_msg.data = list(np.asarray(q_des).flatten())
                    kinova_joint_pub.publish(joint_msg)
                # import pdb; pdb.set_trace()

                # publish new pose command
                # pose_msg = PoseStamped()
                # pose_msg.pose = ComposePoseFromTransQuat(curr_pose)
                # kinova_control_pub.publish(pose_msg)

                print(np.linalg.norm(gripper_err))
                if np.linalg.norm(gripper_err) < 1e-2:
                    state = "grasp"
                    curr_pose = np.hstack((robot_pts[-1], [0, 0.707, 0.707, 0]))
                    curr_pose[2] = -0.22
            except Exception as e:
                # print(e)
                pass

        elif state == "grasp":
            print("current: " + str(robot_pts[-1]))
            print("desired: " + str(curr_pose[:3]))
            print()
            if np.linalg.norm(np.array(curr_pose[:3]) - robot_pts[-1]) < 1e-2:
                break

            q_des = run_ik(list(curr_pose))
            if q_des is not None:
                q_des = np.array(q_des)*180/math.pi # convert to degrees
                joint_msg = Float64MultiArray()
                joint_msg.data = list(np.asarray(q_des).flatten())
                kinova_joint_pub.publish(joint_msg)

        r.sleep()
