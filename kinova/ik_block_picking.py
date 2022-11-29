from __future__ import division
from __future__ import print_function

import math
from collections import deque
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from ar_track_alvar_msgs.msg import AlvarMarkers
import time

from utils import close_gripper, open_gripper, publish_pose

class BlockPickingTask():
    def __init__(self):
        self.robot_pts = deque()
        self.buffer_size = 10
        self.robot_joint_state = []
        self.ar_marker_ids = [4, 5, 6] # all markers of interest (the ones that are on blocks)
        self.ar_markers = {id: None for id in self.ar_marker_ids}
        self.ar_markers_wrist = {id: None for id in self.ar_marker_ids}
        self.current_marker = self.ar_marker_ids[0]
        # self.marker_xyz = []
        # self.marker_wrist_xyz = []
        self.desired_pose = [0.3, -0.4, -0.1, 0, 0.707, 0.707, 0] # TOOL_POS from siemens demo

        # for visual servoing with ar tag
        self.desired_xy = np.array([0.01, 0.06])
        self.Kp = 1.4*np.eye(2)

        self.state = "sense"

        # TODO: read orientation from /kinova/pose_tool_in_base_fk
        self.position_sub = rospy.Subscriber("/kinova/current_position", Point, self.robot_cb, queue_size=1)
        self.joint_sub = rospy.Subscriber("/kinova/current_joint_state", Float64MultiArray, self.joint_cb, queue_size=1)
        self.ar_tag_base_sub = rospy.Subscriber("ar_marker_status", AlvarMarkers, self.ar_cb, queue_size=1)
        self.ar_tag_wrist_sub = rospy.Subscriber("ar_marker_wrist_status", AlvarMarkers, self.ar_wrist_cb, queue_size=1)

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
                self.ar_markers[marker_id] = None
                continue
            marker_pos = msg.markers[marker_idx].pose.pose.position
            xy_pos = [marker_pos.x, marker_pos.y]
            self.ar_markers[marker_id] = xy_pos
    
    def ar_wrist_cb(self, msg):
        if len(msg.markers) == 0:
            return
        msg_idxs = {msg.markers[i].id: i for i in range(len(msg.markers))} # mapping from marker_id -> idx in msg.markers list
        for marker_id in self.ar_marker_ids:
            try:
                marker_idx = msg_idxs[marker_id]
            except:
                # this marker id wasn't found in the message, set its position to None
                self.ar_markers_wrist[marker_id] = None
                continue
            marker_pos = msg.markers[marker_idx].pose.pose.position
            xy_pos = [marker_pos.x, marker_pos.y]
            self.ar_markers_wrist[marker_id] = xy_pos

    def reached_desired(self):
        # TODO: compare orientation as well (use quaternion dot product & check for negative)
        return (np.linalg.norm(np.array(self.desired_pose[:3]) - self.robot_pts[-1]) <= 1e-2)

    def run(self):
        time.sleep(0.5)
        if self.state == "sense":
            current_marker = None
            for marker in self.ar_marker_ids:
                # if y coord < -0.1, it's outside the workspace
                if (self.ar_markers[marker] is not None) and (self.ar_markers[marker][1] >= -0.1):
                    current_marker = marker
                    break
            if current_marker is None:
                print("No blocks detected.")
                self.state = "done"
            else:
                self.current_marker = current_marker
                self.desired_pose[:2] = self.ar_markers[self.current_marker]
                self.desired_pose[2] = -0.1
                print("finished sensing")
                self.state = "move_to_block"
        elif self.state == "move_to_block":
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type="joint")
            if self.reached_desired():
                print("finished moving")
                self.state = "sense_visual_servoing"
        elif self.state == "sense_visual_servoing":
            while self.ar_markers_wrist[self.current_marker] is None:
                pass
            print("finished sensing for servoing")
            self.state = "visual_servoing"
        elif self.state == "visual_servoing":
            # xy = self.marker_wrist_xyz[:2]
            xy = self.ar_markers_wrist[self.current_marker]
            err = xy - self.desired_xy
            # +y in gripper -> +x in pose of marker
            # +x in gripper -> +y in pose of marker
            # translate this error to a new pose command (coordinate system is different, flip x & y)
            gripper_err = np.array([-err[1], -err[0]])
            desired_pose = np.hstack((self.robot_pts[-1], [0, 0.707, 0.707, 0]))
            desired_pose[:2] = (self.Kp.dot(gripper_err)) + desired_pose[:2]
            self.desired_pose = list(desired_pose)

            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type="joint")
            if self.reached_desired():
                print("finished servoing")
                self.state = "lower_gripper"
                self.desired_pose[:3] = self.robot_pts[-1]
                self.desired_pose[2] = -0.22
        elif self.state == "lower_gripper":
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type="joint")
            if self.reached_desired():
                print("finished lowering")
                self.state = "grasp_block"
        elif self.state == "grasp_block":
            close_gripper()
            print("finished grasping block")
            self.state = "move_block"
        elif self.state == "move_block":
            self.desired_pose[:3] = [0.3, -0.5, 0.0] # position of box/general drop location
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type="joint")
            if self.reached_desired():
                print("finished moving block")
                self.state = "lower_block"
                self.desired_pose[2] = -0.1 # want to only set desired pose once
        elif self.state == "lower_block":
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type="joint")
            if self.reached_desired():
                print("finished lowering block")
                self.state = "drop_block"
        elif self.state == "drop_block":
            open_gripper()
            print("finished dropping block")
            self.state = "return_home"
        elif self.state == "return_home":
            self.desired_pose = [0.3, -0.4, 0.2, 0, 0.707, 0.707, 0]
            publish_pose(self.desired_pose, self.robot_joint_state, cmd_type="joint")
            if self.reached_desired():
                print("returned home")
                self.state = "sense"
        elif self.state == "done":
            return "done"


if __name__ == "__main__":
    rospy.init_node("mfi_demo")

    task = BlockPickingTask()

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if task.run() == "done":
            break
        r.sleep()
