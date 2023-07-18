import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2 as cv

from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_utils import overlay_timesteps

def use_camera():
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

h_pos_buf = []
def mouse_move(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        h_pos_buf.append([x, y])
        if len(h_pos_buf) > 5:
            h_pos_buf.pop(0)

def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

if __name__ == "__main__":
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