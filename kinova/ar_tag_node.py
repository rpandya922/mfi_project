from __future__ import division
from __future__ import print_function

from collections import deque
import rospy
from std_msgs.msg import Float64MultiArray
import tf
import time

ar_markers = [4, 5, 6]
marker_pts = [deque() for _ in ar_markers]
buffer_size = 10

if __name__ == "__main__":
    rospy.init_node("ar_tracking")
    r = rospy.Rate(2)

    listener = tf.TransformListener()
    remove_counter = [0 for _ in ar_markers] # used to detect if marker is no longer visible (presumably moved by human)

    ar_marker_pub = rospy.Publisher("ar_marker_status", Float64MultiArray, queue_size=1)

    time.sleep(2)
    while not rospy.is_shutdown():

        for marker_i, marker in enumerate(ar_markers):                    
            marker_topic = "ar_marker_" + str(marker)
            # if marker_topic not in frames_dict.keys():
            #     print marker_topic + " not found"
            try:
                (trans, rot) = listener.lookupTransform("/world", "/" + marker_topic, rospy.Time(0))
                block_pose = [0.375, -0.259, -0.1, 0, 0.707, 0.707, 0]
                block_pose[:2] = trans[:2]
                block_pose[0] = block_pose[0] + 0.02
                block_pose[1] = block_pose[1] + 0.02
                marker_pts[marker_i].append(block_pose)
                # TODO: make this smarter
                marker_msg = Float64MultiArray()
                msg_data = [marker]
                msg_data.extend(block_pose[:3])
                marker_msg.data = msg_data
                ar_marker_pub.publish(marker_msg)
            except (tf.ExtrapolationException, tf.LookupException) as e:
                # print "lookup exception"
                remove_counter[marker_i] += 1
            except Exception as e:
                # print e
                pass
        # print remove_counter

        # NOTE: slightly unstable
        # check if we need to remove any markers
        to_delete = [i for i in range(len(remove_counter)) if (remove_counter[i] > 9)]
        ar_markers_new = []
        remove_counter_new = []
        marker_pts_new = []
        for marker_i, marker in enumerate(ar_markers):
            if marker_i in to_delete:
                print("marker " + str(marker) + " not seen for 10 steps")
                continue
            ar_markers_new.append(marker)
            remove_counter_new.append(remove_counter[marker_i])
            marker_pts_new.append(marker_pts[marker_i])
        ar_markers = ar_markers_new
        remove_counter = remove_counter_new
        marker_pts = marker_pts_new

        r.sleep()