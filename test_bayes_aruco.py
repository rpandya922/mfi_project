import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pyrealsense2 as rs

from bayes_inf import BayesEstimatorAR
from cbp_model import BetaBayesEstimatorAR
from detect_human_aruco import HumanWristDynamics
from intention_utils import overlay_timesteps

def test_cv2_cam():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    h_pos_buf = []
    buf_size = 10

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, _ = frame.shape
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        parameters = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        frame_markers = cv.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

        if ids is not None:
            marker_id = ids[0][0]
            if marker_id != 0:
                continue
            marker_corners = corners[0][0]
            marker_center = np.mean(marker_corners, axis=0)
            xy_pos = (marker_center / np.array([width, height]) * 22) - 11 # give some extra space
            h_pos_buf.append(xy_pos)
            while len(h_pos_buf) > buf_size:
                h_pos_buf.pop(0)
        if len(h_pos_buf) > 0:
            print(h_pos_buf[-1])

        # Display the resulting frame
        cv.imshow('frame', frame_markers)
        if cv.waitKey(1) == ord('q'):
            break

def test_rs2_cam():
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                # images = np.hstack((color_image, depth_colormap))
                images = color_image

            frame = color_image
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
            parameters = cv.aruco.DetectorParameters()
            detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
            frame_markers = cv.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

            if ids is not None:
                marker_id = ids[0][0]
                if marker_id != 0:
                    continue
                marker_corners = corners[0][0]
                marker_center = np.mean(marker_corners, axis=0)
                print(marker_center)

            # Show images
            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.imshow('RealSense', images)
            cv.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()

def get_marker_center(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    frame_markers = cv.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    if ids is not None:
        marker_id = ids[0][0]
        if marker_id != 0:
            return None
        marker_corners = corners[0][0]
        marker_center = np.mean(marker_corners, axis=0)
        return marker_center
    return None

def get_h_pos(marker_center, top_right, bottom_right, bottom_left, top_left):
    # map pixel coordinates to x,y coordinates
    x = marker_center[0]
    y = marker_center[1]
    xh = np.zeros((2, 1))
    xh[1] = (x - top_right[0]) / (bottom_right[0] - top_right[0]) * 22 - 11
    xh[0] = (y - top_right[1]) / (bottom_left[1] - top_right[1]) * 22 - 11
    return -xh

def bayes_inf_rs2():
    """
    Bayesian inference on human wrist position using realsense camera
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("No RGB feed found.")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # create bayesian estimator
    np.random.seed(0)
    h_dynamics = HumanWristDynamics(0.1)
    n_goals = 3
    goals = np.random.uniform(size=(4, n_goals))*20 - 10
    goals[[1,3],:] = np.zeros((2, 3))
    belief = BayesEstimatorAR(goals, h_dynamics, beta=0.5)
    beta_belief = BetaBayesEstimatorAR(goals, betas=[1e-3, 1e-2, 1e-1, 1e0], dynamics=h_dynamics)
    beta_belief.actions = belief.actions

    xh = None

    # corners
    top_right = np.array([294, 55])
    bottom_right = np.array([533, 83])
    bottom_left = np.array([574, 328])
    top_left = np.array([212, 284])

    # data storing
    xh_traj = np.zeros((4, 0))
    beliefs = np.zeros((0, n_goals))
    beliefs_beta = np.zeros((beta_belief.belief.shape[0], beta_belief.belief.shape[1], 0))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
    axes = axes.flatten()
    ax = axes[0]
    belief_ax = axes[1]
    beta_belief_ax = axes[2]
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

    loop_idx = 0
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            frame = np.asanyarray(color_frame.get_data())

            marker_center = get_marker_center(frame)

            if marker_center is not None:
                xh_next = get_h_pos(marker_center, top_right, bottom_right, bottom_left, top_left)
                if xh is None:
                    xh = np.array([xh_next[0], [0], xh_next[1], [0]])
                uh = (xh_next - xh[[0,2]])
            else:
                if xh is None:
                    continue
                uh = np.zeros((2,1))
            
            # update belief
            if np.linalg.norm(uh) > 1e-4: # only if action is non-zero            
                belief.update_belief(xh, uh)
                beta_belief.update_belief(xh, uh)

            # step dynamics forward
            xh = h_dynamics.step(xh, uh)

            # save data
            xh_traj = np.hstack((xh_traj, xh))
            beliefs = np.vstack((beliefs, belief.belief))
            beliefs_beta = np.dstack((beliefs_beta, beta_belief.belief))

            if np.linalg.norm(goals - xh, axis=0).min() < 1:
                belief.actions = belief.actions_w_zero
            else:
                belief.actions = belief.actions_wo_zero
            beta_belief.actions = belief.actions

            ax.cla()
            overlay_timesteps(ax, xh_traj, [], n_steps=loop_idx)
            ax.scatter(xh[0], xh[2], c="blue", s=100)
            ax.scatter(goals[0], goals[2], c=goal_colors)
            ax.set_xlim(-11, 11)
            ax.set_ylim(-11, 11)
            ax.set_aspect('equal')

            belief_ax.cla()
            for goal_idx in range(n_goals):
                belief_ax.plot(beliefs[:,goal_idx], c=goal_colors[goal_idx])

            beta_belief_ax.cla()
            for beta_idx in range(beta_belief.betas.shape[0]):
                beta_belief_ax.plot(beliefs_beta[:,beta_idx,:].sum(axis=0), label=f"b={beta_belief.betas[beta_idx]}")
            beta_belief_ax.legend()

            plt.pause(0.001)

            # Show images
            # cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            # cv.imshow('RealSense', frame)
            # cv.waitKey(1)

            loop_idx += 1
    finally:
        # Stop streaming
        pipeline.stop()



if __name__ == "__main__":
    # test_rs2_cam()
    bayes_inf_rs2()
