import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# [OUT OF DATE]
# 0: nothing/broken
# 1: front camera depth(?) map
# 2: front camera normal rgb
# 3: nothing/broken
# 4: top camera depth(?) map
# 5: top camera normal rgb
# 6: nothing/broken
# 7: top camera depth(?) map
# 8: top camera normal rgb
# 9: nothing/broken
# 10: nothing/broken
# 11: nothing/broken
# 12: nothing/broken
# 13: side camera depth(?) map (looks green)
# 14: side camera normal rgb

front_vid = cv2.VideoCapture(int(sys.argv[1]))
front_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
front_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

front_writer = cv2.VideoWriter('demo_front4.avi', 
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