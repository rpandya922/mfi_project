
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# define a video capture object

front_vid = cv2.VideoCapture(11)
front_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
front_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

front_writer = cv2.VideoWriter('demo_front3.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            30, (1920, 1080))

side_vid = cv2.VideoCapture(13)
side_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
side_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

side_writer = cv2.VideoWriter('demo_side3.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            30, (1920, 1080))

while(True):
      
    # Capture the video frame
    # by frame
    side_ret, side_frame = side_vid.read()
    front_ret, front_frame = front_vid.read()
    if(not side_ret or not front_ret):
        print(front_ret)
        continue
    # Display the resulting frame
    cv2.imshow('frame', side_frame)
    front_writer.write(front_frame)
    side_writer.write(side_frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
side_vid.release()
front_vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
