from collections import deque
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Point
import tf
import time

robot_pts = deque()
buffer_size = 10

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
    desired_xy = np.array([0.0, 0.9])
    Kp = np.eye(2)

    r = rospy.Rate(1)
    time.sleep(2)
    while not rospy.is_shutdown():
        
        if len(robot_pts) < buffer_size:
            continue

        try:
            (trans, rot) = listener.lookupTransform("/camera_link", "/ar_marker_" + str(marker))

            xy = np.array(trans[:2])
            err = xy - desired_xy

            # TODO: translate this error to a new pose command (coordinate system is different, flip x & y)
        except:
            pass

        r.sleep()