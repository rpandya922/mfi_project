from __future__ import division
from __future__ import print_function

from collections import deque
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point
import tf
import time

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

def robot_callback(msg):
    global robot_pts, buffer_size
    try:
        ee_pt = np.array([msg.x, msg.y, msg.z]) # end effector location
        robot_pts.append(ee_pt)
        while len(robot_pts) > buffer_size:
            robot_pts.popleft()
    except Exception:
        pass

if __name__=="__main__":
    rospy.init_node("speed_test")

    # ros topics
    kinova_control_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=1)
    kinova_gripper_sub = rospy.Subscriber("/kinova/current_position", Point, robot_callback, queue_size=1)
    listener = tf.TransformListener()

    # temporary
    marker = 5

    # PD controller
    prev_err = 0
    Kp = 4*np.eye(3)
    Kd = 0.3*np.eye(3)

    r = rospy.Rate(10)
    dt = 1 / 10
    time.sleep(2)
    box_pose = [0.3, -0.5, 0.0, 0, 0.707, 0.707, 0]
    goal_pose = [0.5, 0.0, 0.0, 0, 0.707, 0.707, 0]
    print("started")
    while not rospy.is_shutdown():

        if len(robot_pts) < buffer_size:
            continue

        # pose_cmd = np.hstack((robot_pts[-1], [0, 0.707, 0.707, 0]))
        err = robot_pts[-1] - goal_pose[:3]
        d_err = (err - prev_err) / dt
        pose_cmd = np.hstack((robot_pts[-1], [0, 0.707, 0.707, 0]))
        pose_cmd[:3] = -Kp.dot(err) - Kd.dot(d_err) #+ pose_cmd[:3]

        print(np.linalg.norm(err)) # find out why SS error is 0.1 with Kd=0

        if np.linalg.norm(np.array(goal_pose[:3]) - robot_pts[-1]) < 1e-3:
            break

        # publish new pose command
        pose_msg = PoseStamped()
        pose_msg.pose = ComposePoseFromTransQuat(pose_cmd)
        kinova_control_pub.publish(pose_msg)

        r.sleep()