#!/usr/bin/python
import rospy
from geometry_msgs.msg import PoseStamped, Pose

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

if __name__=="__main__":
    rospy.init_node("mfi_demo")
    kinova_control_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=1)

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        face_pose1 = [0.3, -0.4, 0.2, 0, 0.526, 0.850, 0]
        face_pose2 = [0.3, -0.4, 0.2, 0, -0.943, -0.332, 0]
        pose_msg = PoseStamped()
        # pose = pose_msg.pose
        # pose.position.x = 0.496
        # pose.position.y = 0.138
        # pose.position.z = 0.5
        # pose.orientation.w = 1.0
        # pose.orientation.x = 0.0
        # pose.orientation.y = 0.0
        # pose.orientation.z = 0.0
        pose_msg.pose = ComposePoseFromTransQuat(face_pose1)

        kinova_control_pub.publish(pose_msg)

        print(pose_msg)

        r.sleep()