from __future__ import division
from __future__ import print_function

from collections import deque
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_openpose_rs2_msgs.msg import HPoseSync
import tf
import time

human_wrist_pts = deque()
buffer_size = 10

def robot_pose_transform(msg):
    global listener, robot_pose_pub
    try:
        output_pose = listener.transformPose("world", msg)
    except:
        return

    robot_pose_pub.publish(output_pose)

def human_intention_pred(msg):
    global human_wrist_pts, buffer_size, human_intention_pub
    dt = 0.1 # running at 10 Hz

    if len(human_wrist_pts) < buffer_size:
        return

    # TODO: add a lock?
    wrist_pos = np.array(human_wrist_pts[-1][:])
    wrist_positions = np.array(human_wrist_pts)

    wrist_vels = []
    # for i in range(buffer_size-1):
    for i in [7, 8]:
        wrist_i = wrist_positions[i]
        wrist_i1 = wrist_positions[i+1]
        wrist_vel = (wrist_i1 - wrist_i) / dt
        wrist_vels.append(wrist_vel)
    avg_vel = np.mean(wrist_vels, axis=0)
    
    marker_locs = []
    marker_ids = []

    for marker in msg.markers:
        if marker.id == 8: # skip marker 8, it's for calibration
            continue
        marker_position = marker.pose.pose.position
        marker_xyz = [marker_position.x, marker_position.y, marker_position.z]
        marker_locs.append(marker_xyz)
        marker_ids.append(marker.id)
        # just print locations of markers & human wrist position
    #     print("marker " + str(marker.id) + " location: " + str(marker_xyz))
    # print("wrist location: " + str(wrist_pos))
    # print()
    
    marker_locs = np.array(marker_locs)
    # dists = np.linalg.norm(marker_locs - wrist_pos, axis=1)
    # print(marker_ids[np.argmin(dists)])

    # # compute theta_v, or angle created by velocity vector
    # theta_v = np.arctan2(avg_vel[1], avg_vel[0])

    # # compute theta between current wrist position and each marker position
    # thetas = []
    # for marker_xyz in marker_locs:
    #     diff = marker_xyz - wrist_positions[buffer_size-1]
    #     thetas.append(np.arctan2(diff[1], diff[0]))
    
    # angle_diffs = [np.abs(theta - theta_v) for theta in thetas]
    # # angle_diffs = [(theta - theta_v) % np.pi for theta in thetas]
    # # print(angle_diffs)
    # # print()
    # print(marker_ids[np.argmin(angle_diffs)])

    # compute unit vector from velocity
    vel_unit = np.array(avg_vel[:2])
    vel_unit = vel_unit / np.linalg.norm(vel_unit)

    # compute unit vectors for direction from wrist position to each marker
    directions = []
    for marker_xyz in marker_locs:
        diff = marker_xyz - wrist_positions[buffer_size-1]
        directions.append(diff[:2] / np.linalg.norm(diff[:2]))
    
    dists = [np.linalg.norm(d - vel_unit) for d in directions]
    intention = marker_ids[np.argmin(dists)]
    # print(intention)

    intention_msg = Float64MultiArray()
    intention_msg.data = [intention]
    human_intention_pub.publish(intention_msg)


def human_pose_cb(msg):
    global human_wrist_pts, openpose_trans, buffer_size
    if msg.body.size != 0:    
        wrist_pt = msg.body.keypoints[4].position
        wrist_pt = np.asarray([[wrist_pt.x], [wrist_pt.y], [wrist_pt.z], [1]])
        wrist_pt = np.matmul(openpose_trans, wrist_pt)
        wrist_pt = [wrist_pt[0,0], wrist_pt[1,0], wrist_pt[2,0]]

        human_wrist_pts.append(wrist_pt)
        while len(human_wrist_pts) > buffer_size:
            human_wrist_pts.popleft()
    
if __name__ == "__main__":
    rospy.init_node("robot_pose_transform")
    r = rospy.Rate(10)

    listener = tf.TransformListener()

    # transforming kinova's end effector position 
    robot_pose_sub = rospy.Subscriber("/kinova/pose_tool_in_base", PoseStamped, robot_pose_transform, queue_size=1)
    robot_pose_pub = rospy.Publisher("/kinova/pose_tool_in_world", PoseStamped, queue_size=1)

    # publishing the ar tag that the human is closest to
    human_pose_sub = rospy.Subscriber("/rs_openpose_3d/human_pose_sync", HPoseSync, human_pose_cb, queue_size=1)
    ar_tag_sub = rospy.Subscriber("ar_marker_in_world", AlvarMarkers, human_intention_pred, queue_size=1)
    openpose_trans = np.matrix([[-0.9956066,  0.0936216,  0.0015747, 0.49533],
                            [0.0899620,  0.9610820, -0.2612055, 0.273142],
                            [-0.0259679, -0.2599162, -0.9652820, 1.57835],
                            [0, 0, 0, 1]])
    human_intention_pub = rospy.Publisher("/human_intention", Float64MultiArray, queue_size=1)


    time.sleep(1)
    while not rospy.is_shutdown():
        r.sleep()
