from __future__ import division
from __future__ import print_function

import sys
import math
from collections import deque
import numpy as np
from scipy import stats
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point, PoseStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_openpose_rs2_msgs.msg import HPoseSync
import time

from utils import close_gripper, open_gripper, publish_pose, publish_joint_state


class BlockPickingTask():
    def __init__(self, mode):
        self.robot_pts = deque()
        self.human_wrist_pts = deque()
        self.human_intention = deque()
        self.buffer_size = 10
        self.robot_joint_state = []
        self.ar_marker_ids = [4, 5, 6] # all markers of interest (the ones that are on blocks)
        self.ar_markers = {id: deque() for id in self.ar_marker_ids}
        self.ar_markers_wrist = {id: deque() for id in self.ar_marker_ids}
        self.current_marker = self.ar_marker_ids[0]
        self.desired_pose = [0.3, -0.4, -0.1, 0, 0.707, 0.707, 0] # TOOL_POS from siemens demo
        # self.home_pose = [33.35, 34.65, 212.46, 272.44, 340.09, 296.42, 160.11]
        self.home_position = [339.39, 24.35, 183.52, 231.28, 356.95, 333.01, 75.52]
        self.desired_joint_position = self.home_position

        # for visual servoing with ar tag
        self.desired_xy = np.array([0.01, 0.06])
        self.Kp = 1.4*np.eye(2)

        self.state = "startup"
        self.cmd_type = "joint"

        # TODO: read orientation from /kinova/pose_tool_in_base_fk
        self.position_sub = rospy.Subscriber("/kinova/current_position", Point, self.robot_cb, queue_size=1)
        self.joint_sub = rospy.Subscriber("/kinova/current_joint_state", Float64MultiArray, self.joint_cb, queue_size=1)
        self.ar_tag_base_sub = rospy.Subscriber("ar_marker_status", AlvarMarkers, self.ar_cb, queue_size=1)
        self.ar_tag_wrist_sub = rospy.Subscriber("ar_marker_wrist_status", AlvarMarkers, self.ar_wrist_cb, queue_size=1)
        self.ssa_enable_pub = rospy.Publisher("/siemens_demo/ssa_enable", Float64MultiArray, queue_size=1)
        self.human_pose_sub = rospy.Subscriber("/rs_openpose_3d/human_pose_sync", HPoseSync, self.human_cb, queue_size=1)
        self.human_intention_sub = rospy.Subscriber("/human_intention", Float64MultiArray, self.intention_cb, queue_size=1)

        # mode 1
        if mode == 1:
            self.ssa = False
            self.mode = "naive"
        elif mode == 2:
            # mode 2
            self.ssa = True
            self.mode = "naive"
        elif mode == 3:
            # mode 3
            self.ssa = False
            self.mode = "proactive"
        elif mode == 4:
            # mode 4
            self.ssa = True
            self.mode = "proactive"

    # read naive intention prediction (from velocity vector)
    def intention_cb(self, msg):
        intention = msg.data[0]
        self.human_intention.append(intention)
        while len(self.human_intention) > self.buffer_size:
            self.human_intention.popleft()

    def human_cb(self, msg):
        try:
            wrist_pt = msg.body.keypoints[4].position
            wrist_pt = np.array([wrist_pt.x, wrist_pt.y, wrist_pt.z, 1])[:,None]
            # TODO: properly load in the openpose matrices for later data processing
            # wrist_pts.append(np.matmul(openpose_trans, wrist_pt).flatten())
            self.human_wrist_pts.append(wrist_pt)
            while len(self.human_wrist_pts) > self.buffer_size:
                self.human_wrist_pts.popleft()
        except Exception as e:
            pass

    def robot_cb(self, msg):
        try:
            ee_pt = np.array([msg.x, msg.y, msg.z])
            self.robot_pts.append(ee_pt)
            while len(self.robot_pts) > self.buffer_size:
                self.robot_pts.popleft()
        except Exception:
            pass
    
    def joint_cb(self, msg):
        try:
            self.robot_joint_state = list(msg.data)
        except:
            pass

    def ar_cb(self, msg):
        if len(msg.markers) == 0:
            return
        msg_idxs = {msg.markers[i].id: i for i in range(len(msg.markers))} # mapping from marker_id -> idx in msg.markers list
        for marker_id in self.ar_marker_ids:
            try:
                marker_idx = msg_idxs[marker_id]
            except:
                # this marker id wasn't found in the message, set its position to None
                self.ar_markers[marker_id].append(None)
                while len(self.ar_markers[marker_id]) > self.buffer_size:
                    self.ar_markers[marker_id].popleft()
                continue
            marker_pos = msg.markers[marker_idx].pose.pose.position
            xy_pos = [marker_pos.x, marker_pos.y]
            self.ar_markers[marker_id].append(xy_pos)
            while len(self.ar_markers[marker_id]) > self.buffer_size:
                self.ar_markers[marker_id].popleft()
    
    def ar_wrist_cb(self, msg):
        if len(msg.markers) == 0:
            return
        msg_idxs = {msg.markers[i].id: i for i in range(len(msg.markers))} # mapping from marker_id -> idx in msg.markers list
        for marker_id in self.ar_marker_ids:
            try:
                marker_idx = msg_idxs[marker_id]
            except:
                # this marker id wasn't found in the message, set its position to None
                self.ar_markers_wrist[marker_id].append(None)
                while len(self.ar_markers_wrist[marker_id]) > self.buffer_size:
                    self.ar_markers_wrist[marker_id].popleft()
                continue
            marker_pos = msg.markers[marker_idx].pose.pose.position
            xy_pos = [marker_pos.x, marker_pos.y]
            self.ar_markers_wrist[marker_id].append(xy_pos)
            while len(self.ar_markers_wrist[marker_id]) > self.buffer_size:
                self.ar_markers_wrist[marker_id].popleft()

    def reached_desired(self):
        # TODO: compare orientation as well (use quaternion dot product & check for negative)
        return (np.linalg.norm(np.array(self.desired_pose[:3]) - self.robot_pts[-1]) <= 1e-2)

    def reached_desired_easy(self):
        # TODO: compare orientation as well (use quaternion dot product & check for negative)
        return (np.linalg.norm(np.array(self.desired_pose[:3]) - self.robot_pts[-1]) <= 5e-2)

    def reached_desired_joint(self):
        return (np.linalg.norm(np.array(self.robot_joint_state) - np.array(self.desired_joint_position)) < 1)

    def is_deque_none(self, d):
        all_none = True
        for e in d:
            if e is not None:
                all_none = False
        return all_none

    def get_last_position(self, d):
        last_position = None
        for i in range(len(d)-1, -1, -1):
            e = d[i]
            if e is None:
                continue
            else:
                return e

    def choose_marker(self):
        current_marker = None
        if self.mode == "proactive":
            time.sleep(1)
            # h_intention = stats.mode(self.human_intention)[0][0]
            h_intention = self.human_intention[-1]
            print(h_intention)
            for marker in self.ar_marker_ids:
                # if y coord < -0.1, it's outside the workspace
                # if (self.ar_markers[marker] is not None) and (self.ar_markers[marker][1] >= -0.1) and (marker != h_intention):
                #     current_marker = marker
                #     break
                if (not self.is_deque_none(self.ar_markers[marker])) and (self.get_last_position(self.ar_markers[marker])[1] >= -0.1) and (marker != h_intention):
                    current_marker = marker
                    break
        elif self.mode == "naive":
            for marker in self.ar_marker_ids:
                # if y coord < -0.1, it's outside the workspace
                # if (self.ar_markers[marker] is not None) and (self.ar_markers[marker][1] >= -0.1):
                #     current_marker = marker
                #     break
                if (not self.is_deque_none(self.ar_markers[marker])) and (self.get_last_position(self.ar_markers[marker])[1] >= -0.1):
                    current_marker = marker
                    break
        else:
            return
        return current_marker

    def run(self):
        # print(self.ar_markers)
        if self.ssa and (len(self.human_wrist_pts) < self.buffer_size):
            print("Waiting for human pose")
            return

        ssa_msg = Float64MultiArray()
        if self.ssa:
            ssa_msg.data = [1]
        else:
            ssa_msg.data = [0]
        self.ssa_enable_pub.publish(ssa_msg)

        if self.state == "startup":
            self.desired_joint_position = self.home_position
            publish_joint_state(self.desired_joint_position)

            if self.reached_desired_joint():
                self.state = "sense"
        elif self.state == "sense":
            current_marker = self.choose_marker()
            if current_marker is None:
                print("No blocks detected.")
                self.state = "done"
            else:
                self.current_marker = current_marker
                marker_position = self.get_last_position(self.ar_markers[self.current_marker])
                self.desired_pose[:2] = marker_position
                self.desired_pose[0] = 0.45
                self.desired_pose[2] = -0.1
                print("finished sensing")
                self.state = "move_behind_block"
        # TODO: add a position behind block to move towards first
        elif self.state == "move_behind_block":
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type=self.cmd_type)
            if self.reached_desired_easy():
                marker_position = self.get_last_position(self.ar_markers[self.current_marker])
                self.desired_pose[:2] = marker_position
                self.desired_pose[2] = -0.1
                print("finished moving behind block")
                self.state = "move_to_block"
        elif self.state == "move_to_block":
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type=self.cmd_type)
            if self.reached_desired():
                print("finished moving")
                self.state = "sense_visual_servoing"
        elif self.state == "sense_visual_servoing":
            # while self.ar_markers_wrist[self.current_marker] is None:
            #     pass
            while self.is_deque_none(self.ar_markers_wrist[self.current_marker]):
                pass
            # always disable SSA after sensing for servoing so robot won't hit the table
            self.ssa = False
            print("finished sensing for servoing")
            self.state = "visual_servoing"
        elif self.state == "visual_servoing":
            # xy = self.marker_wrist_xyz[:2]
            xy = self.get_last_position(self.ar_markers_wrist[self.current_marker])
            if xy is None:
                self.state = "sense"
                return
            err = xy - self.desired_xy
            # +y in gripper -> +x in pose of marker
            # +x in gripper -> +y in pose of marker
            # translate this error to a new pose command (coordinate system is different, flip x & y)
            gripper_err = np.array([-err[1], -err[0]])
            desired_pose = np.hstack((self.robot_pts[-1], [0, 0.707, 0.707, 0]))
            desired_pose[:2] = (self.Kp.dot(gripper_err)) + desired_pose[:2]
            self.desired_pose = list(desired_pose)

            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type=self.cmd_type)
            if self.reached_desired():
                print("finished servoing")
                self.state = "lower_gripper"
                self.desired_pose[:3] = self.robot_pts[-1]
                self.desired_pose[2] = -0.22 #-0.18
        elif self.state == "lower_gripper":
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type=self.cmd_type)
            if self.reached_desired():
                print("finished lowering")
                self.state = "grasp_block"
        elif self.state == "grasp_block":
            close_gripper()
            print("finished grasping block")
            self.state = "move_block"
        elif self.state == "move_block":
            self.desired_pose[:3] = [0.3, -0.5, 0.0] # position of box/general drop location
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type=self.cmd_type)
            if self.reached_desired():
                print("finished moving block")
                # self.state = "lower_block"
                self.state = "drop_block"
                self.desired_pose[2] = -0.1 # want to only set desired pose once
        elif self.state == "lower_block":
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type=self.cmd_type)
            if self.reached_desired():
                print("finished lowering block")
                self.state = "drop_block"
        elif self.state == "drop_block":
            open_gripper()
            print("finished dropping block")
            self.state = "return_home"
        elif self.state == "return_home":
            self.desired_pose = [0.3, -0.4, 0.2, 0, 0.707, 0.707, 0]
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type=self.cmd_type)
            if self.reached_desired():
                print("returned home")
                self.state = "sense"
        elif self.state == "done":
            return "done"


if __name__ == "__main__":
    rospy.init_node("mfi_demo")

    task = BlockPickingTask(mode=int(sys.argv[1]))

    r = rospy.Rate(10)
    time.sleep(0.5)
    while not rospy.is_shutdown():
        if task.run() == "done":
            break
        r.sleep()
