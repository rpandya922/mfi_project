import numpy as np
import rosbag
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

if __name__ == "__main__":
    bag = rosbag.Bag('./data/2022-10-11-16-38-35.bag')
    bag_content = bag.read_messages()

    wrist_pts = []
    robot_pts = []
    get_h_pose = False
    get_r_pose = False

    human_topic = "/rs_openpose_3d/human_pose_sync"
    robot_topic = "/kinova/current_position"

    openpose_trans = np.matrix([[-0.9956066,  0.0936216,  0.0015747, 0.49533],
                            [0.0899620,  0.9610820, -0.2612055, 0.273142],
                            [-0.0259679, -0.2599162, -0.9652820, 1.57835],
                            [0, 0, 0, 1]])

    for topic, msg, t in bag_content:
        if topic == human_topic:
            if msg.body.size != 0:    
                wrist_pt = msg.body.keypoints[4].position
                wrist_pt = np.asarray([[wrist_pt.x], [wrist_pt.y], [wrist_pt.z], [1]])
                wrist_pt = np.matmul(openpose_trans, wrist_pt)

                get_h_pose = True
        if topic == robot_topic:
            robot_pt = np.array([msg.x, msg.y, msg.z])

            get_r_pose = True
        
        if get_h_pose and get_r_pose:
            wrist_pts.append([wrist_pt[0,0], wrist_pt[1,0], wrist_pt[2,0]])
            robot_pts.append(robot_pt)
            get_h_pose = False
            get_r_pose = False
    
    bag.close()
    np.savetxt("./data/wrist_pose.txt", np.array(wrist_pts))
    np.savetxt("./data/robot_pose.txt", np.array(robot_pts))