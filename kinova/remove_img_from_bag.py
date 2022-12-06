from __future__ import print_function
from __future__ import division

import rosbag

# need this file to process bag files and remove image data
user_names = ["p4_1"]
modes = [3]
topics_list = ["/rs_openpose_3d/human_pose_sync",
          "/rs_openpose_3d/time_step",
          "/kinova/pose_tool_in_world",
          "/kinova/current_position",
          "/kinova/current_joint_state",
          "ar_marker_status",
          "ar_marker_wrist_status",
          "/siemens_demo/ssa_enable",
          "/rs_openpose_3d/human_pose_sync",
          "/human_intention",
          "ar_marker_in_world",
        #   "/cam_rs2/color/image_raw",
          "/siemens_demo/joint_cmd",
          "/siemens_demo/gripper_cmd"]
for user_name in user_names:
    for mode in modes:
        filepath = "./data/pilot_study/" + user_name + "/mode" + str(mode)
        readbag = rosbag.Bag(filepath + ".bag")

        writebag = rosbag.Bag(filepath + "_small.bag", 'w')

        for topic, msg, t in readbag.read_messages(topics=topics_list):
            # if topic != "/cam_rs2/color/image_raw":
            writebag.write(topic, msg, t)
        
        readbag.close()
        writebag.close()



