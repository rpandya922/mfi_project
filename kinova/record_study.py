#!python3

import sys
import os
import signal
import subprocess
import time
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

user_name = "test"
mode = sys.argv[2]
topics = ["/rs_openpose_3d/human_pose_sync",
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
          "/cam_rs2/color/image_raw",
          "/siemens_demo/joint_cmd",
          "/siemens_demo/gripper_cmd"]

pro = subprocess.Popen(["rosbag", "record"] + topics + ["-O", "./data/pilot_study/" + user_name + "/mode" + mode + ".bag"])

# use command line arg to read video stream (should be /dev/video11, so sys.argv[1] should equal 11)
front_vid = cv2.VideoCapture(int(sys.argv[1]))
front_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
front_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

front_writer = cv2.VideoWriter("./data/pilot_study/" + user_name + "/mode" + mode + ".avi", 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            30, (1920, 1080))

while(True):
      
    # Capture the video frame
    # by frame
    front_ret, front_frame = front_vid.read()
    if(not front_ret):
        continue
    # Display the resulting frame
    cv2.imshow('frame', front_frame)
    front_writer.write(front_frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
front_vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
