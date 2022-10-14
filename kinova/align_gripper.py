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
    rospy.init_node("align_gripper")

    # ros topics
    kinova_control_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=1)
    kinova_gripper_sub = rospy.Subscriber("/kinova/current_position", Point, robot_callback, queue_size=1)
    listener = tf.TransformListener()

    # temporary
    marker = 5

    # visual servoing with ar tag
    desired_xy = np.array([0.01, 0.05])
    Kp = 0.8*np.eye(2)

    state = "align"
    r = rospy.Rate(1)
    time.sleep(2)
    while not rospy.is_shutdown():
        
        if len(robot_pts) < buffer_size:
            continue

        try:
            (trans, rot) = listener.lookupTransform("/camera_color_frame", "/ar_marker_" + str(marker), rospy.Time(0))

            xy = np.array(trans[:2])
            err = xy - desired_xy
            # +y in gripper -> +x in pose of marker
            # +x in gripper -> +y in pose of marker
            # translate this error to a new pose command (coordinate system is different, flip x & y)
            gripper_err = np.array([-err[1], -err[0]])

            curr_pose = np.hstack((robot_pts[-1], [0, 0.707, 0.707, 0]))
            curr_pose[:2] = (Kp.dot(gripper_err)) + curr_pose[:2]
            # import pdb; pdb.set_trace()

            # publish new pose command
            pose_msg = PoseStamped()
            pose_msg.pose = ComposePoseFromTransQuat(curr_pose)
            kinova_control_pub.publish(pose_msg)
            # print pose_msg.pose
            # print (Kp.dot(gripper_err))
            if np.linalg.norm(Kp.dot(gripper_err)) < 1e-3:
                state = "grasp"
        except Exception as e:
            # print e
            pass
            
        if state == "grasp":
            curr_pose = np.hstack((robot_pts[-1], [0, 0.707, 0.707, 0]))
            curr_pose[2] = -0.22

            if np.linalg.norm(np.array(curr_pose[:3]) - robot_pts[-1]) < 1e-3:
                break

            # publish new pose command
            pose_msg = PoseStamped()
            pose_msg.pose = ComposePoseFromTransQuat(curr_pose)
            kinova_control_pub.publish(pose_msg)

        r.sleep()