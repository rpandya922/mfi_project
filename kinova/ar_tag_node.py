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

# ar_markers = []
# ar_markers_wrist = []

# def ar_marker_cb(msg):
#     global ar_markers, ar_markers_wrist
#     # if msg.markers[i].header.frame_id == "/cam_rs2_color_frame", it's being seen in the table camera
#     # if it equals "/camera_color_frame", it's being seen in the arm's wrist camera
#     if (len(msg.markers) > 0) and (msg.markers[0].header.frame_id == "/cam_rs2_color_frame"):
#         ar_markers = [marker.id for marker in msg.markers]
#     if (len(msg.markers) > 0) and (msg.markers[0].header.frame_id == "/camera_color_frame"):
#         ar_markers_wrist = [marker.id for marker in msg.markers]

def ar_marker_cb2(msg):
    global listener, ar_marker_pub, ar_marker_wrist_pub
    # if msg.markers[i].header.frame_id == "/cam_rs2_color_frame", it's being seen in the table camera
    # if it equals "/camera_color_frame", it's being seen in the arm's wrist camera
    if (len(msg.markers) > 0) and (msg.markers[0].header.frame_id == "/cam_rs2_color_frame"):
        publisher = ar_marker_pub
        reference_frame = "/kinova_base"
    elif (len(msg.markers) > 0) and (msg.markers[0].header.frame_id == "/camera_color_frame"):
        publisher = ar_marker_wrist_pub
        reference_frame = "/camera_color_frame"
    else:
        return

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
    publisher.publish(msg)

if __name__ == "__main__":
    rospy.init_node("ar_tracking")
    r = rospy.Rate(2)

    listener = tf.TransformListener()
    # remove_counter = [0 for _ in ar_markers] # used to detect if marker is no longer visible (presumably moved by human)

    ar_marker_pub = rospy.Publisher("ar_marker_status", AlvarMarkers, queue_size=1)
    ar_marker_wrist_pub = rospy.Publisher("ar_marker_wrist_status", AlvarMarkers, queue_size=1)
    ar_marker_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, ar_marker_cb2, queue_size=1)

    time.sleep(2)
    while not rospy.is_shutdown():

        # marker_msgs = []
        # for marker_i, marker in enumerate(ar_markers):
        #     marker_topic = "ar_marker_" + str(marker)
        #     try:
        #         (trans, rot) = listener.lookupTransform("/kinova_base", "/" + marker_topic, rospy.Time(0))
        #         # block_pose = [0.375, -0.259, -0.1, 0, 0.707, 0.707, 0]
        #         # block_pose[:3] = trans[:3]
        #         # block_pose[0] = block_pose[0] + 0.02
        #         # block_pose[1] = block_pose[1] + 0.02

        #         # TODO: make this smarter
        #         # marker_msg = Float64MultiArray()
        #         # msg_data = [marker]
        #         # msg_data.extend(block_pose[:3])
        #         # marker_msg.data = msg_data
        #         # ar_marker_pub.publish(marker_msg)
        #         # marker_poses.append((trans, rot))
        #         rot_wxyz = [rot[1], rot[2], rot[3], rot[0]]
        #         transquat = trans + rot_wxyz # concat lists
        #         marker_pose = PoseStamped()
        #         marker_pose.pose = ComposePoseFromTransQuat(transquat)

        #         marker_msg = AlvarMarker()
        #         marker_msg.id = marker
        #         marker_msg.pose = marker_pose

        #         marker_msgs.append(marker_msg)
                
        #     except (tf.ExtrapolationException, tf.LookupException) as e:
        #         # print "lookup exception"
        #         pass
        #     except Exception as e:
        #         # print e
        #         pass
        
        # # markers_msg = AlvarMarkers()
        # # markers_msg.markers = marker_msgs
        # # ar_marker_pub.publish(markers_msg)

        # for marker_i, marker in enumerate(ar_markers_wrist):                    
        #     marker_topic = "ar_marker_" + str(marker)
        #     try:
        #         (trans, rot) = listener.lookupTransform("/camera_color_frame", "/" + marker_topic, rospy.Time(0))

        #         # TODO: make this smarter
        #         marker_msg = Float64MultiArray()
        #         msg_data = [marker]
        #         msg_data.extend(trans[:3])
        #         marker_msg.data = msg_data
        #         ar_marker_wrist_pub.publish(marker_msg)
        #     except (tf.ExtrapolationException, tf.LookupException) as e:
        #         # print "lookup exception"
        #         pass
        #     except Exception as e:
        #         # print e
        #         pass

        r.sleep()