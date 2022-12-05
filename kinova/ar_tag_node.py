from __future__ import division
from __future__ import print_function

from collections import deque
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from ar_track_alvar_msgs.msg import AlvarMarkers, AlvarMarker
import tf
import time

from utils import ComposePoseFromTransQuat

def ar_marker_cb2(msg):
    global listener, ar_marker_pub, ar_marker_wrist_pub, ar_marker_world_pub
    # if msg.markers[i].header.frame_id == "/cam_rs2_color_frame", it's being seen in the table camera
    # if it equals "/camera_color_frame", it's being seen in the arm's wrist camera
    if (len(msg.markers) > 0) and (msg.markers[0].header.frame_id == "/cam_rs2_color_frame"):
        publisher = ar_marker_pub
        reference_frame = "/kinova_base"
        publish_in_world = True
    elif (len(msg.markers) > 0) and (msg.markers[0].header.frame_id == "/camera_color_frame"):
        publisher = ar_marker_wrist_pub
        reference_frame = "/camera_color_frame"
        publish_in_world = False
    else:
        return

    world_msg = AlvarMarkers()
    world_msg.markers = [AlvarMarker() for _ in range(len(msg.markers))]
    for i in range(len(msg.markers)):
        marker_id = msg.markers[i].id
        marker_topic = "ar_marker_" + str(marker_id)
        try:
            (trans, rot) = listener.lookupTransform(reference_frame, "/" + marker_topic, rospy.Time(0))
        except:
            return
        rot_wxyz = [rot[1], rot[2], rot[3], rot[0]]
        transquat = trans + rot_wxyz # concat lists
        marker_pose = PoseStamped()
        marker_pose.pose = ComposePoseFromTransQuat(transquat)

        msg.markers[i].pose = marker_pose

        if publish_in_world:
            try:
                (trans_world, rot_world) = listener.lookupTransform("world", "/" + marker_topic, rospy.Time(0))
            except:
                return
            rot_wxyz_world = [rot_world[1], rot_world[2], rot_world[3], rot_world[0]]
            transquat_world = trans_world + rot_wxyz_world # concat lists
            marker_pose_world = PoseStamped()
            marker_pose_world.pose = ComposePoseFromTransQuat(transquat_world)
            world_msg.markers[i].pose = marker_pose_world
            world_msg.markers[i].id = marker_id
    publisher.publish(msg)
    if publish_in_world:
        ar_marker_world_pub.publish(world_msg)

if __name__ == "__main__":
    rospy.init_node("ar_tracking")
    r = rospy.Rate(10)

    listener = tf.TransformListener()

    ar_marker_pub = rospy.Publisher("ar_marker_status", AlvarMarkers, queue_size=1)
    ar_marker_wrist_pub = rospy.Publisher("ar_marker_wrist_status", AlvarMarkers, queue_size=1)
    ar_marker_world_pub = rospy.Publisher("ar_marker_in_world", AlvarMarkers, queue_size=1)
    ar_marker_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, ar_marker_cb2, queue_size=1)

    time.sleep(2)
    while not rospy.is_shutdown():
        r.sleep()