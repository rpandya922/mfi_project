from collections import deque
import os
import numpy as np
import rospy
from std_msgs.msg import Header, Float64MultiArray
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import transforms3d.quaternions as quat
import transforms3d.affines as aff
from ros_openpose_rs2_msgs.msg import HPoseSync
from ar_track_alvar_msgs.msg import AlvarMarkers
import tf
import time

from game_state import GameState

wrist_pts = deque()
robot_pts = deque()
buffer_size = 10
ar_tf = np.eye(4)
can_pos = deque()

static_tf = np.eye(4)
static_tf[0:,3] = np.array([0.02, 0.02, 0.0, 1.0])


def quat2mat_homogeneous(q):
    R = quat.quat2mat([q.w, q.x, q.y, q.z])
    R2 = np.zeros((4, 4))
    R2[:3,:3] = R
    R2[3,3] = 1
    return R2

def PoseStamped_2_mat(p):
    q = p.pose.orientation
    pos = p.pose.position
    T = quat2mat_homogeneous(q)
    T[:3,3] = np.array([pos.x,pos.y,pos.z])
    return T

def Mat_2_posestamped(m,f_id="test"):
    q = quat.mat2quat(m)
    p = PoseStamped(header = Header(frame_id=f_id), #robot.get_planning_frame()
                    pose=Pose(position=Point(*m[:3,3]), 
                    orientation=Quaternion(*q)))
    return p

def T_inv(T_in):
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))

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

def ar_callback(msg):
    markers = msg.markers

    marker_8 = max(markers, key=lambda x: x.id==8).pose
    marker_6 = max(markers, key=lambda x: x.id==6).pose

    Tw1 = PoseStamped_2_mat(marker_8)
    Tw2 = PoseStamped_2_mat(marker_6)
    T2w = T_inv(Tw2)
    T21 = np.matmul(T2w, Tw1) 

    T8_inv = T_inv(static_tf)

    # print position of ar tag in the kinova base frame
    marker_6_pos = marker_6.pose.position
    m6_coord = np.array([marker_6_pos.x, marker_6_pos.y, marker_6_pos.z, 1.0])
    m6_coord2 =  T8_inv.dot(T21).dot(m6_coord)

    # print m6_coord2


# xyz: [0.375, -0.259, 0] wxyz: [0, 0.707, 0.707, 0]
if __name__=="__main__":

    # TODO: make these a set of command line arguments
    use_openpose = False

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
    kinova_grasp_pub = rospy.Publisher("/siemens_demo/gripper_cmd", Float64MultiArray, queue_size=1)
    kinova_gripper_sub = rospy.Subscriber("/kinova/current_position", Point, robot_callback, queue_size=1)
    if use_openpose:
        body_sub = rospy.Subscriber("/rs_openpose_3d/human_pose_sync", HPoseSync, body_callback, queue_size=1)

    ar_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, ar_callback, queue_size=1)

    listener = tf.TransformListener()

    game = GameState()
    curr_action = "sense"
    r = rospy.Rate(10)
    desired_pose = []
    pose_reached = False
    while not rospy.is_shutdown():

        if curr_action == "sense":
            try:
                (trans, rot) = listener.lookupTransform("/kinova_base", "/ar_marker_6", rospy.Time(0))
                can_pose = [0.375, -0.259, 0, 0, 0.707, 0.707, 0]
                can_pose[:2] = trans[:2]
                can_pos.append(can_pose)
            except:
                pass
            if len(can_pos) >= buffer_size:
                curr_action = "nav"

        # publishing messages for actions
        if curr_action == "nav":
            can_pose = can_pos[-1]
            desired_pose = can_pose
            pose_msg = PoseStamped()
            pose_msg.pose = ComposePoseFromTransQuat(can_pose)
            kinova_control_pub.publish(pose_msg)
            if pose_reached:
                curr_action = "lower"
                pose_reached = False
                desired_pose = []
        elif curr_action == "lower":
            can_pose = can_pos[-1]
            can_pose[2] = -0.12
            desired_pose = can_pose
            pose_msg = PoseStamped()
            pose_msg.pose = ComposePoseFromTransQuat(can_pose)
            kinova_control_pub.publish(pose_msg)
            if pose_reached:
                curr_action = "grasp"
                pose_reached = False
                desired_pose = []
        elif curr_action == "grasp":
            g_cmd = Float64MultiArray()
            g_cmd.data = [0]
            kinova_grasp_pub.publish(g_cmd)
            curr_action = "raise"
        elif curr_action == "raise":
            time.sleep(0.5)
            can_pose = can_pos[-1]
            can_pose[2] = 0.2
            desired_pose = can_pose
            pose_msg = PoseStamped()
            pose_msg.pose = ComposePoseFromTransQuat(can_pose)
            kinova_control_pub.publish(pose_msg)
            if pose_reached:
                curr_action = "none"
                pose_reached = False
                desired_pose = []
        elif curr_action == "none":
            break
        
        # checking if desired position is reached
        if len(robot_pts) > 0 and desired_pose != []:
            if np.linalg.norm(np.array(desired_pose[:3]) - robot_pts[-1]) < 0.01:
                pose_reached = True

        if len(wrist_pts) < buffer_size:
            # rospy.loginfo("waiting for queue data")
            continue
        # print(wrist_pts[-1])

        r.sleep()