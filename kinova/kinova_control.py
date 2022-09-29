from collections import deque
import os
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point
import transforms3d.quaternions as quat
import transforms3d.affines as aff
from ros_openpose_rs2_msgs.msg import HPoseSync

from game_state import GameState

wrist_pts = deque()
robot_pts = deque()
buffer_size = 10

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

def body_callback(msg):
    global wrist_pts, buffer_size
    try:
        # rospy.loginfo_throttle(5.0, "Got body.")
        wrist_pt = msg.body.keypoints[4].position
        wrist_pt = np.array([wrist_pt.x, wrist_pt.y, wrist_pt.z, 1])[:,None]
        wrist_pts.append(np.matmul(openpose_trans, wrist_pt).flatten())
        while len(wrist_pts) > buffer_size:
            wrist_pts.popleft()
    except Exception:
        pass

def robot_callback(msg):
    global robot_pts, buffer_size
    try:
        ee_pt = np.array([msg.x, msg.y, msg.z]) # end effector location
        robot_pts.append(ee_pt)
        while len(robot_pts) > buffer_size:
            robot_pts.popleft()
    except Exception:
        pass

def ComposeAffine(tx, ty, tz, w, rx, ry, rz, Z=np.ones(3)):
    R = quat.quat2mat([w, rx, ry, rz])
    T = [tx, ty, tz]
    mat = aff.compose(T, R, Z)
    return mat

def ReadCameraExtrinsic(cam_param_path, serial):

    infile = os.path.join(cam_param_path, serial, "extrinsics.txt")

    if not os.path.exists(infile):
        return np.eye(4)

    with open(infile, 'r') as f_extrinsic:

        trans_line = f_extrinsic.readline()
        words = trans_line.split(' ')
        tx = float(words[2])
        ty = float(words[3])
        tz = float(words[4])

        q_line = f_extrinsic.readline()
        words = q_line.split(' ')
        w  = float(words[2])
        rx = float(words[3])
        ry = float(words[4])
        rz = float(words[5])
    
    mat = ComposeAffine(tx, ty, tz, w, rx, ry, rz)

    return mat
# xyz: [0.375, -0.259, 0] wxyz: [0, 0.707, 0.707, 0]
if __name__=="__main__":
    rospy.init_node("mfi_demo")

    # cam matrices
    cam_param_path = '/home/ruic/Documents/meta_cobot_wksp/src/meta_cobot_learning/hri_tasks/calibration'
    serial_rs2 = '908212070729'
    object_trans = ReadCameraExtrinsic(cam_param_path, serial_rs2)
    serial_rs0 = '845112071853'
    openpose_trans = ReadCameraExtrinsic(cam_param_path, serial_rs0)
    object_intrinsic_inv = np.linalg.inv(np.matrix([[1315.26001, 0, 973.856262],
                                                [0, 1316.05347, 547.485962], 
                                                [0, 0, 1]]))

    kinova_control_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=1)
    kinova_gripper_sub = rospy.Subscriber("/kinova/current_position", Point, robot_callback, queue_size=1)
    body_sub = rospy.Subscriber("/rs_openpose_3d/human_pose_sync", HPoseSync, body_callback, queue_size=1)

    game = GameState()
    curr_action = "nav"
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        # face_pose1 = [0.3, -0.4, 0.2, 0, 0.526, 0.850, 0]
        # face_pose2 = [0.3, -0.4, 0.2, 0, -0.943, -0.332, 0]
        can_pose = [0.375, -0.259, 0, 0, 0.707, 0.707, 0]
        pose_msg = PoseStamped()
        pose_msg.pose = ComposePoseFromTransQuat(can_pose)

        if curr_action == "nav":
            kinova_control_pub.publish(pose_msg)
        elif curr_action == "lower":
            pose_msg.pose.position.z = 0.2
            kinova_control_pub.publish(pose_msg)
        
        if len(robot_pts) > 0:
            if np.linalg.norm(np.array(can_pose[:3]) - robot_pts[-1]) < 0.1:
                curr_action = "lower"

        if len(wrist_pts) < buffer_size:
            # rospy.loginfo("waiting for queue data")
            continue
        # print(wrist_pts[-1])

        r.sleep()