import numpy as np
import matplotlib.pyplot as plt

from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from human import Human
from robot import Robot
from intention_utils import overlay_timesteps

if __name__ == "__main__":
    horizon = 100
    ts = 0.05
    np.random.seed(6)
    # randomly initialize xh0, xr0, goals
    # xh0 = np.random.uniform(size=(4, 1))*20 - 10
    # xh0[[1,3]] = np.zeros((2, 1))
    # xr0 = np.random.uniform(size=(4, 1))*20 - 10
    # xr0[[1,3]] = np.zeros((2, 1))

    # goals = np.random.uniform(size=(4, 3))*20 - 10
    # goals[[1,3],:] = np.zeros((2, 3))
    # r_goal = goals[:,[np.random.randint(0,3)]]
    xh0 = np.array([[0, 0.0, -5, 0.0]]).T
    xr0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T

    goals = np.array([
        [5.0, 0.0, 0.0, 0.0],
        [-5.0, 0.0, 5.0, 0.0],
        [5.0, 0.0, 5.0, 0.0],
    ]).T
    r_goal = goals[:,[0]]

    dynamics_h = DIDynamics(ts)
    dynamics_r = DIDynamics(ts)
    belief = BayesEstimator(thetas=goals, dynamics=dynamics_r, beta=1)
    human = BayesHuman(xh0, dynamics_h, goals, belief, gamma=20)
    # human = Human(xh0, dynamics_h, goals, gamma=10)
    robot = Robot(xr0, dynamics_r, r_goal)

    xh_traj = np.zeros((4, horizon))
    xr_traj = np.zeros((4, horizon))
    h_goals = np.zeros((4, horizon))
    h_goal_reached = np.zeros((1, horizon))

    fig, ax = plt.subplots()

    for i in range(horizon):
        # plot data
        ax.cla()
        # plot traj so far
        overlay_timesteps(ax, xh_traj[:,:i], xr_traj[:,:i], goals, n_steps=i)
        ax.scatter(human.x[0], human.x[2], c="b")
        ax.scatter(robot.x[0], robot.x[2], c="r")
        # ell = Ellipse(xy=(xh0[0],xh0[2]),
        #             width=6, height=6,
        #             fill=False, color='r')
        # ax.add_patch(ell)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)

        # save data
        xh_traj[:,[i]] = human.x
        xr_traj[:,[i]] = robot.x
        h_goal = human.get_goal()
        h_goals[:,[i]] = h_goal
        # check if human reached its goal
        if np.linalg.norm(human.x - h_goal) < 0.1:
            h_goal_reached[:,i] = 1

        # take step
        uh = human.get_u(robot.x)
        if i == 0:
            ur = robot.get_u(human.x, robot.x, human.x)
        else:
            ur = robot.get_u(human.x, xr_traj[:,[i-1]], xh_traj[:,[i-1]])

        # update human's belief (if applicable)
        if type(human) == BayesHuman:
            human.update_belief(robot.x, ur)

        xh = human.step(uh)
        xr = robot.step(ur)