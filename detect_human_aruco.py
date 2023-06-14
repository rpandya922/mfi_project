import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    frame_markers = cv.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    if ids is not None:
        import ipdb; ipdb.set_trace()
    # Display the resulting frame
    cv.imshow('frame', frame_markers)
    if cv.waitKey(1) == ord('q'):
        break