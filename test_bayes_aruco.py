import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import cv2 as cv
import pyrealsense2 as rs

from dynamics import DIDynamics
from robot import Robot
from human import ARHuman
from bayes_inf import BayesEstimatorAR
from cbp_model import CBPEstimatorAR, BetaBayesEstimatorAR
from detect_human_aruco import HumanWristDynamics
from intention_utils import overlay_timesteps
from generate_path_loop_multiple_obs import generate_trajectory_belief

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

def get_homography(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    frame_markers = cv.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    id_to_xy = {1: np.array([-10, 10]), 2: np.array([10, 10]), 3: np.array([-10, -10]), 4: np.array([10, -10])}

    print(ids)

    if (ids is not None) and (1 in ids) and (2 in ids) and (3 in ids) and (4 in ids):
        source_xy = []
        dest_xy = []
        for idx, marker_id in enumerate(ids):
            marker_center = np.mean(corners[idx][0], axis=0)
            xy = id_to_xy[marker_id[0]]
            source_xy.append(marker_center)
            dest_xy.append(xy)
        H, _ = cv.findHomography(np.array(source_xy), np.array(dest_xy))
        h, w = frame.shape[:2]
        frame_markers = cv.warpPerspective(frame, H, (w, h))
        cv.imshow('RealSense', frame_markers)
        cv.waitKey(1)

        import ipdb; ipdb.set_trace()
        return H

def get_h_pos(marker_center, top_right, bottom_right, bottom_left, top_left):
    # map pixel coordinates to x,y coordinates
    x = marker_center[0]
    y = marker_center[1]
    xh = np.zeros((2, 1))
    xh[1] = (x - top_right[0]) / (bottom_right[0] - top_right[0]) * 22 - 11
    xh[0] = (y - top_right[1]) / (bottom_left[1] - top_right[1]) * 22 - 11
    return -xh

def min_goal_dists(goals):
    # return the minimum distance between any pair of goals
    dists = []
    for i in range(goals.shape[1]):
        for j in range(goals.shape[1]):
            if i == j:
                continue
            dists.append(np.linalg.norm(goals[:,i] - goals[:,j]))
    return np.min(dists)

def bayes_inf_rs2(robot_type="cbp"):
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
    ts = 0.1
    traj_horizon = 20
    n_goals = 3
    plot_mode = "debug" # alternatively "study"

    xr = np.random.uniform(-10, 10, (4, 1))
    xr[[1,3]] = 0
    xh0 = np.zeros((4,1))
    goals = np.random.uniform(size=(4, n_goals))*20 - 10
    goals[[1,3],:] = np.zeros((2, 3))
    while min_goal_dists(goals) < 5:
        goals = np.random.uniform(size=(4, n_goals))*20 - 10
        goals[[1,3],:] = np.zeros((2, 3))
    r_goal = goals[:,[2]] # this is arbitrary since it'll be changed in simulations later anyways

    h_dynamics = HumanWristDynamics(ts)
    r_dynamics = DIDynamics(ts=ts)

    human = ARHuman(xh0, h_dynamics, goals)
    robot = Robot(xr, r_dynamics, r_goal, dmin=3)

    r_belief = CBPEstimatorAR(thetas=goals, dynamics=h_dynamics, beta=0.5)
    belief_nominal = BayesEstimatorAR(goals, h_dynamics, beta=0.5)
    # beta_belief = BetaBayesEstimatorAR(goals, betas=[1e-3, 1e-2, 1e-1, 1e0], dynamics=h_dynamics)
    beta_belief = BetaBayesEstimatorAR(goals, betas=[1e-2, 1e0], dynamics=h_dynamics)
    beta_belief.actions = belief_nominal.actions

    xh = None

    # corners
    top_right = np.array([294, 55])
    bottom_right = np.array([533, 83])
    bottom_left = np.array([574, 328])
    top_left = np.array([212, 284])

    # data storing
    xh_traj = np.zeros((4, 0))
    xr_traj = np.zeros((4, 0))
    r_beliefs = np.zeros((0, n_goals))
    r_beliefs_cond = np.zeros((n_goals, n_goals, 0))
    beliefs_nominal = np.zeros((0, n_goals))
    beliefs_beta = np.zeros((beta_belief.belief.shape[0], beta_belief.belief.shape[1], 0))
    r_goal_idxs = []
    is_robot_waiting = []
    is_human_waiting = []

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
    axes = axes.flatten()
    ax = axes[0]
    belief_ax = axes[1]
    beta_belief_ax = axes[2]
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

    both_at_goal_count = 0

    loop_idx = 0
    try:
        robot_wait_time = 5
        h_goal_reached = False
        r_goal_reached = False
        while True:

            # check if both agents have waited at their goals for long enough. if so, resample 2 new goals and reset beliefs
            if both_at_goal_count >= 10:
                # resample goals
                # find goal human is at
                h_goal_idx = np.argmin(np.linalg.norm(xh - goals, axis=0))
                r_goal_idx = np.argmin(np.linalg.norm(xr - goals, axis=0))
                
                new_goals = np.random.uniform(size=(4, 2))*20 - 10
                new_goals[[1,3],:] = np.zeros((2, 2))
                goals[:,[h_goal_idx, r_goal_idx]] = new_goals
                while min_goal_dists(goals) < 5:
                    new_goals = np.random.uniform(size=(4, 2))*20 - 10
                    new_goals[[1,3],:] = np.zeros((2, 2))
                    goals[:,[h_goal_idx, r_goal_idx]] = new_goals
                
                both_at_goal_count = 0

                # reset beliefs
                r_belief.belief = np.ones(n_goals) / n_goals
                r_belief.thetas = goals
                belief_nominal.belief = np.ones(n_goals) / n_goals
                belief_nominal.thetas = goals
                beta_belief.belief = np.ones((n_goals, len(beta_belief.betas))) / (n_goals*len(beta_belief.betas))
                beta_belief.thetas = goals

            # Wait for a coherent color frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            frame = np.asanyarray(color_frame.get_data())

            # get_homography(frame)

            marker_center = get_marker_center(frame)

            if marker_center is not None:
                xh_next = get_h_pos(marker_center, top_right, bottom_right, bottom_left, top_left)
                if xh is None:
                    xh = np.array([xh_next[0], [0], xh_next[1], [0]])
                    human.x = xh
                uh = (xh_next - xh[[0,2]])
            else:
                if xh is None:
                    continue
                uh = np.zeros((2,1)) + 0.05*(2*np.random.rand(2,1)-1)
            
            # update belief
            if np.linalg.norm(uh) > 1e-4: # only if action is non-zero            
                r_belief_prior = belief_nominal.update_belief(xh, uh)
                beta_belief.update_belief(xh, uh)
            else:
                r_belief_prior = belief_nominal.belief

            human_waiting = False
            if (beta_belief.betas[np.argmax(beta_belief.belief, axis=1)] < 1e0).all():
                human_waiting = True
            is_human_waiting.append(human_waiting)
            # print(human_waiting)

            # TODO: add "influence" goal selection objective when robot is not waiting
            influence_human = False
            if human_waiting:
                influence_human = True

            if not human_waiting and robot_wait_time > 0 and robot_type == "cbp":
                ur = np.zeros((2,1))
                obs_loc = None
                robot_wait_time -= 1
                is_robot_waiting.append(True)
            else:
                robot_wait_time = 0 # make sure robot commits to waiting only at the start
                # generate safe trajectory for robot
                safety, ur_traj, obs_loc = generate_trajectory_belief(robot, human, r_belief, traj_horizon, goals, plot=False)
                # if robot is sufficiently close to its goal, just move straight to the goal without considering human
                if (np.linalg.norm(robot.x[[0,2]] - robot.goal[[0,2]]) < 1.5) or h_goal_reached:
                    ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
                else:
                    ur = ur_traj[:,[0]]
                is_robot_waiting.append(False)

            xh_next = human.dynamics.A @ human.x + human.dynamics.B @ uh
            # loop through goals and compute belief update for each potential robot goal
            posts = []
            for goal_idx in range(goals.shape[1]):
                goal = goals[:,[goal_idx]]
                # compute CBP belief update
                r_belief_post = r_belief.weight_by_score(r_belief_prior, goal, xh_next, beta=0.5)
                posts.append(r_belief_post)
            
            # pick goal with highest probability on human's most likely goal
            goal_scores = []
            for goal_idx in range(goals.shape[1]):
                p = posts[goal_idx]
                h_belief_score = p[np.argmax(r_belief_prior)]
                goal_change_score = -0.01*np.linalg.norm(goals[:,[goal_idx]] - robot.goal)
                # goal_change_score = 0
                goal_scores.append(h_belief_score + goal_change_score)
            goal_idx = np.argmax(goal_scores)

            if robot_type == "baseline":
                goal_idx = np.linalg.norm(robot.x - goals, axis=0).argmin()
            elif robot_type == "baseline_belief":
                # pick the closest goal that the human is not moving towards
                dists = np.linalg.norm(robot.x - goals, axis=0)
                dists[belief_nominal.belief.argmax()] = np.inf
                goal_idx = dists.argmin()

            robot.goal = goals[:,[goal_idx]]

            # update robot's belief
            r_belief_post = posts[goal_idx]
            r_belief.belief = r_belief_post

            # step dynamics forward
            xh = h_dynamics.step(xh, uh)
            human.x = xh
            xr = robot.step(ur)

            # check how long human and robot have been at goals
            h_at_goal = False
            r_at_goal = False
            if np.linalg.norm(goals - xh, axis=0).min() <= 0.5:
                h_at_goal = True
            if np.linalg.norm(goals - xr, axis=0).min() <= 0.5:
                r_at_goal = True
            if h_at_goal and r_at_goal:
                both_at_goal_count += 1
            else:
                both_at_goal_count = 0

            # save data
            xh_traj = np.hstack((xh_traj, xh))
            xr_traj = np.hstack((xr_traj, xr))
            r_beliefs = np.vstack((r_beliefs, r_belief.belief))
            beliefs_nominal = np.vstack((beliefs_nominal, belief_nominal.belief))
            beliefs_beta = np.dstack((beliefs_beta, beta_belief.belief))
            # save robot's actual intended goal
            r_goal_idxs.append(np.argmin(np.linalg.norm(robot.goal - goals, axis=0)))

            if np.linalg.norm(goals - xh, axis=0).min() < 1:
                belief_nominal.actions = belief_nominal.actions_w_zero
            else:
                belief_nominal.actions = belief_nominal.actions_wo_zero
            beta_belief.actions = belief_nominal.actions

            ax.cla()
            overlay_timesteps(ax, xh_traj, xr_traj, n_steps=loop_idx)
            ax.scatter(xh[0], xh[2], c="blue", s=100)
            ax.scatter(xr[0], xr[2], c="red", s=100)
            ax.scatter(goals[0], goals[2], c=goal_colors)
            ax.set_xlim(-11, 11)
            ax.set_ylim(-11, 11)
            ax.set_aspect('equal')

            # plot circle indicating percent time waited at goal
            if both_at_goal_count > 0:
                # find goal human is at
                h_goal_idx = np.argmin(np.linalg.norm(xh - goals, axis=0))
                circle_xy = goals[:,[h_goal_idx]][[0,2]]
                r_goal_idx = np.argmin(np.linalg.norm(xr - goals, axis=0))
                r_circle_xy = goals[:,[r_goal_idx]][[0,2]]
                radius = 1
                theta1 = 90 # start circle at top
                theta2 = (360)*(both_at_goal_count / 10) + 90
                theta2 = min(360+90, theta2)
                arc = Arc(circle_xy.flatten(), radius*2, radius*2, color='g', lw=3, theta1=theta1, theta2=theta2)
                ax.add_patch(arc)
                arc = Arc(r_circle_xy.flatten(), radius*2, radius*2, color='g', lw=3, theta1=theta1, theta2=theta2)
                ax.add_patch(arc)

            belief_ax.cla()
            # plot nominal belief in dotted lines
            for goal_idx in range(n_goals):
                belief_ax.plot(beliefs_nominal[:,goal_idx], c=goal_colors[goal_idx], linestyle="--")
            # plot conditional belief for chosen goal in solid line
            for goal_idx in range(n_goals):
                belief_ax.plot(r_beliefs[:,goal_idx], c=goal_colors[goal_idx])

            beta_belief_ax.cla()
            for beta_idx in range(beta_belief.betas.shape[0]):
                beta_belief_ax.plot(beliefs_beta[:,beta_idx,:].sum(axis=0), label=f"b={beta_belief.betas[beta_idx]}")
            beta_belief_ax.legend()

            plt.pause(0.001)

            # Show images
            cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
            cv.imshow('RealSense', frame)
            cv.waitKey(1)

            loop_idx += 1
    finally:
        # Stop streaming
        pipeline.stop()



if __name__ == "__main__":
    # test_rs2_cam()
    bayes_inf_rs2()
