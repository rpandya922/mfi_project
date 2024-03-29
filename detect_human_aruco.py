import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2 as cv
import control
from scipy.special import softmax
import os

from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_utils import overlay_timesteps
from cbp_model import CBPEstimator, BetaBayesEstimator

class HumanWristDynamics(object):
    def __init__(self, ts):
        self.ts = ts
        self.A = np.eye(4)
        # direct position control
        self.B = np.array([[1, 0],
                           [0, 0],
                           [0, 1],
                           [0, 0]])
        self.gamma = 0 # no collision avoidance control
        self.n = 4
        self.m = 2

    def step(self, x, u):
        return self.A @ x + self.B @ u
    
    def get_goal_control(self, x, goal):
        diff = (goal - x)[[0,2]]
        # compute unit vector in this direction
        diff_unit = diff / np.linalg.norm(diff)
        # action is scaled version of unit vector
        return 2.0*diff_unit

    def get_robot_control(self, x, xr):
        return np.zeros((2,1))

def get_human_pos(frame):
    height, width, _ = frame.shape
    box_size = int(height * 0.75) # number of pixels 
    height_offset = int((height - box_size) / 2)
    width_offset = int((width - box_size) / 2)
    
    box_left = width_offset
    box_right = width_offset + box_size
    box_top = height_offset
    box_bottom = height_offset + box_size

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    # frame_markers = cv.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    if ids is not None:
        marker_id = ids[0][0]
        marker_corners = corners[0][0]
        marker_center = np.mean(marker_corners, axis=0)

        # map marker_center to xy position [-10, 10] within box
        marker_center = marker_center - np.array([box_left, box_top])
        xy_pos = (marker_center / np.array([box_size, box_size]) * 20) - 10
        # clip to [-10, 10]
        xy_pos = np.clip(xy_pos, -10, 10)
        # xy_pos = (marker_center / np.array([width, height]) * 24) - 12 # give some extra space

        return -xy_pos

def use_camera():
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
        # cv.imshow('frame', frame_markers)
        if cv.waitKey(1) == ord('q'):
            break

h_pos_buf = []
def mouse_move(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        h_pos_buf.append([x, y])
        if len(h_pos_buf) > 5:
            h_pos_buf.pop(0)

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

def play_game_mouse():
    T = 200
    ts = 0.05

    np.random.seed(4)
    # randomly initialize xh0, xr0, goals
    xh = np.random.uniform(size=(4, 1))*20 - 10
    xh[[1,3]] = np.zeros((2, 1))
    xr = np.random.uniform(size=(4, 1))*20 - 10
    xr[[1,3]] = np.zeros((2, 1))

    goals = np.random.uniform(size=(4, 3))*20 - 10
    goals[[1,3],:] = np.zeros((2, 3))
    r_goal = goals[:,[np.random.randint(0,3)]]

    dynamics_h = DIDynamics(ts)
    dynamics_r = DIDynamics(ts)

    belief = BayesEstimator(thetas=goals, dynamics=dynamics_r, beta=1)
    human = BayesHuman(xh, dynamics_h, goals, belief, gamma=1)
    robot = Robot(xr, dynamics_r, r_goal)

    xh_traj = np.zeros((4, T))
    xr_traj = np.zeros((4, T))
    h_goal_idxs = np.zeros((1, T))
    h_goal_reached = np.zeros((1, T))
    xh_traj[:,[0]] = xh
    xr_traj[:,[0]] = xr

    fig, ax = plt.subplots()
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
    plt.connect('motion_notify_event', mouse_move)
    img_paths = ["./data/img/person.png", "./data/img/robot.png"]
    goal_paths = ["./data/img/diamond.png", "./data/img/diamond.png", "./data/img/diamond.png"]

    for i in range(T):
        # plot
        ax.cla()
        overlay_timesteps(ax, xh_traj[:,0:i], xr_traj[:,0:i], n_steps=i)

        # plot goals as diamonds
        for j in range(goals.shape[1]):
            ab = AnnotationBbox(getImage(goal_paths[j], zoom=0.5), (goals[0,j], goals[2,j]), frameon=False)
            ax.add_artist(ab)
        # ax.scatter(goals[0], goals[2], c=goal_colors, s=100)

        # plot human and robot images
        ab = AnnotationBbox(getImage(img_paths[0], zoom=0.1), (human.x[0], human.x[2]), frameon=False)
        ax.add_artist(ab)
        ab = AnnotationBbox(getImage(img_paths[1], zoom=0.1), (robot.x[0], robot.x[2]), frameon=False)
        ax.add_artist(ab)
        # ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
        # ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)

        # get mouse position on matplotlib plot
        # mouse_pos = plt.ginput(1)
        # print(mouse_pos)
        if len(h_pos_buf) > 0:
            # mouse_pos = np.mean(h_pos_buf, axis=0)
            # human.goal = np.array([[mouse_pos[0]], [0], [mouse_pos[1]], [0]])
            mouse_pos = np.array(h_pos_buf[-1])
            human.x = np.array([[mouse_pos[0]], [0], [mouse_pos[1]], [0]])

        # h_goal = human.get_goal()
        # h_goal_idxs[:,[i]] = np.argmin(np.linalg.norm(goals - h_goal, axis=0))
        # check if human reached its goal
        # if np.linalg.norm(human.x - h_goal) < 0.1:
        #     h_goal_reached[:,i] = 1

        # take step
        # uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # update human's belief (if applicable)
        if type(human) == BayesHuman:
            human.update_belief(robot.x, ur)

        # xh = human.step(uh)
        xr = robot.step(ur)

        # save data
        xh_traj[:,[i]] = human.x
        xr_traj[:,[i]] = robot.x
    print("done")

def pick_robot_goal(human, goals, r_posts, r_belief_nominal, r_belief_beta):
    highest_betas = []
    for goal_idx in range(goals.shape[1]):
        g_belief = r_belief_beta.belief[goal_idx,:]
        highest_betas.append(r_belief_beta.betas[np.argmax(g_belief)])
    # check if nominal belief is close to uniform
    if np.linalg.norm(r_belief_nominal.belief - np.ones(r_belief_nominal.belief.shape[0]) / r_belief_nominal.belief.shape[0]) < 0.1:
        # check highest beta per goal
        if max(highest_betas) <= 0.1: # robot should pick a goal to influence person towards
            # print("influence")
            # compute closest goal to the human
            goal_dists = np.linalg.norm(goals - human.x, axis=0)
            min_idx = np.argmin(goal_dists)
            r_goal_idx = np.argmax([p[min_idx] for p in r_posts])
            return  r_goal_idx, False
        else:
            # robot should wait
            # print("waiting")
            return 0, True
    # belief is not close to uniform, robot picks goal that keeps human's goal unchanged
    r_belief_prior = r_belief_nominal.belief
    goal_idx = np.argmax([p[np.argmax(r_belief_prior)] for p in r_posts])
    return goal_idx, False

def play_game_camera(debug=True, save_video=False):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    h_pos_buf = []
    buf_size = 10
    T = 200
    ts = 0.1

    np.random.seed(4)
    # randomly initialize xh0, xr0, goals
    xh = np.random.uniform(size=(4, 1))*20 - 10
    xh[[1,3]] = np.zeros((2, 1))
    xr = np.random.uniform(size=(4, 1))*20 - 10
    xr[[1,3]] = np.zeros((2, 1))

    goals = np.random.uniform(size=(4, 3))*20 - 10
    goals[[1,3],:] = np.zeros((2, 3))
    r_goal = goals[:,[np.random.randint(0,3)]]

    # dynamics_h = DIDynamics(ts)
    dynamics_h = HumanWristDynamics(ts)
    dynamics_r = DIDynamics(ts)
    # make robot move faster to its goal
    K, _, _ = control.dlqr(dynamics_r.A, dynamics_r.B, 30*np.eye(dynamics_r.n), np.eye(dynamics_r.m))
    dynamics_r.K = K

    belief = BayesEstimator(thetas=goals, dynamics=dynamics_r, beta=1)
    human = BayesHuman(xh, dynamics_h, goals, belief, gamma=1)
    
    robot = Robot(xr, dynamics_r, r_goal)
    r_belief = CBPEstimator(thetas=goals, dynamics=dynamics_h, beta=1)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=dynamics_h, beta=1)
    r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[0.01, 0.1, 1, 10, 100], dynamics=dynamics_h)

    r_beliefs = r_belief.belief[:,None]
    r_beliefs_nominal = r_belief_nominal.belief[:,None]
    r_beliefs_beta = r_belief_beta.belief[:,:,None]

    xh_traj = np.zeros((4, T))
    xr_traj = np.zeros((4, T))
    h_goal_idxs = np.zeros((1, T))
    h_goal_reached = np.zeros((1, T))
    xh_traj[:,[0]] = xh
    xr_traj[:,[0]] = xr

    if debug:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,7))
        axes = axes.flatten()
        ax = axes[0]
        r_belief_ax = axes[1]
        r_belief_beta_ax = axes[2]
    else:
        fig, ax = plt.subplots(figsize=(5,5))
    ax.set_aspect('equal', adjustable='box')
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
    img_paths = ["./data/img/person.png", "./data/img/robot.png"]
    goal_paths = ["./data/img/diamond.png", "./data/img/diamond.png", "./data/img/diamond.png"]

    for i in range(T):
        # plot
        ax.cla()
        if i > 100:
            xh_plot = xh_traj[:,i-100:i]
            xr_plot = xr_traj[:,i-100:i]
        else:
            xh_plot = xh_traj[:,0:i]
            xr_plot = xr_traj[:,0:i]
        overlay_timesteps(ax, xh_plot, xr_plot, n_steps=i)

        if debug:
            ax.scatter(goals[0], goals[2], c=goal_colors, s=100)
        else:
            # plot goals as diamonds
            for j in range(goals.shape[1]):
                ab = AnnotationBbox(getImage(goal_paths[j], zoom=0.5), (goals[0,j], goals[2,j]), frameon=False)
                ax.add_artist(ab)

        # plot human and robot images
        ab = AnnotationBbox(getImage(img_paths[0], zoom=0.1), (human.x[0], human.x[2]), frameon=False)
        ax.add_artist(ab)
        ab = AnnotationBbox(getImage(img_paths[1], zoom=0.1), (robot.x[0], robot.x[2]), frameon=False)
        ax.add_artist(ab)
        # ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
        # ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        # plot belief
        if debug:
            r_belief_ax.cla()
            for goal_idx in range(goals.shape[1]):
                r_belief_ax.plot(r_beliefs[goal_idx], c=goal_colors[goal_idx], label=f"P(g{goal_idx})")
                r_belief_ax.plot(r_beliefs_nominal[goal_idx], c=goal_colors[goal_idx], linestyle="--", label=f"P(g{goal_idx})")
            r_belief_ax.legend()

            r_belief_beta_ax.cla()
            # for now, plotting marginalized belief
            for beta_idx in range(r_belief_beta.betas.shape[0]):
                r_belief_beta_ax.plot(r_beliefs_beta[:,beta_idx,:].sum(axis=0), label=f"b={r_belief_beta.betas[beta_idx]}")
            r_belief_beta_ax.legend()

        plt.pause(0.001)

        # get position of human from camera
        # Capture frame-by-frame
        ret, frame = cap.read()
        xy_pos = get_human_pos(frame)
        if save_video:
            # saving camera data
            filepath = f"./data/videos/aruco/frames1/{i:03d}.png"
            # create folders if they don't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            cv.imwrite(filepath, frame)

            # saving PLT data
            filepath = f"./data/videos/aruco/frames2/{i:03d}.png"
            # create folders if they don't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300)

        if xy_pos is not None:
            h_pos_buf.append(xy_pos)
            while len(h_pos_buf) > buf_size:
                h_pos_buf.pop(0)
            xh_new = np.array([[xy_pos[0]], [0], [xy_pos[1]], [0]])
            uh = (xh_new - human.x)[[0,2]]
            # human.x = xh_new
        else:
            uh = np.zeros((2,1))

        # update robot's belief
        r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, robot.x, return_likelihood=True)
        r_belief_prior = r_belief_nominal.belief
        r_belief_beta.belief = r_belief_beta.update_belief_(human.x, uh, robot.x)
        state = human.dynamics.A @ human.x + human.dynamics.B @ uh
        # loop through goals and compute belief update for each
        posts = []
        for goal_idx in range(goals.shape[1]):
            goal = goals[:,[goal_idx]]
            # compute CBP belief update
            r_belief_post = r_belief.weight_by_score(r_belief_prior, goal, state, beta=0.1)
            posts.append(r_belief_post)

        # robot chooses goal
        goal_idx, wait = pick_robot_goal(human, goals, posts, r_belief_nominal, r_belief_beta)
        robot.goal = goals[:,[goal_idx]]

        # update robot's belief
        r_belief_post = posts[goal_idx]
        r_belief.belief = r_belief_post

        # choose robot control
        if wait:
            ur = np.zeros((2,1))
        else:
            ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        xh = human.step(uh)
        xr = robot.step(ur)

        # check if human is at a goal
        goals_changed = False
        goal_dists = np.linalg.norm(goals - human.x, axis=0)
        min_idx = np.argmin(goal_dists)
        if goal_dists[min_idx] < 0.5:
            # resample this goal 
            new_goal = np.random.uniform(size=(4, 1))*20 - 10
            new_goal[[1,3]] = np.zeros((2, 1))
            # make sure new goal is sufficiently far away from two other goals
            other_goals = np.delete(goals, min_idx, axis=1)
            while np.min(np.linalg.norm(other_goals - new_goal, axis=0)) < 5:
                new_goal = np.random.uniform(size=(4, 1))*20 - 10
                new_goal[[1,3]] = np.zeros((2, 1))
            goals[:,[min_idx]] = new_goal
            goals_changed = True
            
        # check if robot is at a goal
        goal_dists = np.linalg.norm(goals - robot.x, axis=0)
        min_idx = np.argmin(goal_dists)
        if goal_dists[min_idx] < 0.5:
            # resample this goal 
            new_goal = np.random.uniform(size=(4, 1))*20 - 10
            new_goal[[1,3]] = np.zeros((2, 1))
            other_goals = np.delete(goals, min_idx, axis=1)
            while np.min(np.linalg.norm(other_goals - new_goal, axis=0)) < 5:
                new_goal = np.random.uniform(size=(4, 1))*20 - 10
                new_goal[[1,3]] = np.zeros((2, 1))
            goals[:,[min_idx]] = new_goal
            goals_changed = True
        
        if goals_changed:
            # update beliefs
            r_belief.thetas = goals
            r_belief.belief = np.ones(goals.shape[1]) / goals.shape[1]

            r_belief_nominal.thetas = goals
            r_belief_nominal.belief = np.ones(goals.shape[1]) / goals.shape[1]

            r_belief_beta.thetas = goals
            n_theta = r_belief_beta.thetas.shape[1]
            n_beta = len(r_belief_beta.betas)
            r_belief_beta.belief = np.ones((n_theta, n_beta)) / (n_theta*n_beta)

        # save data
        xh_traj[:,[i]] = human.x
        xr_traj[:,[i]] = robot.x
        r_beliefs = np.hstack((r_beliefs, r_belief.belief[:,None]))
        r_beliefs_nominal = np.hstack((r_beliefs_nominal, r_belief_nominal.belief[:,None]))
        r_beliefs_beta = np.dstack((r_beliefs_beta, r_belief_beta.belief))

    print("done")

if __name__ == "__main__":
    # use_camera()
    # play_game_mouse()
    play_game_camera(debug=True, save_video=False)