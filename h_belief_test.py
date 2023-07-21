import numpy as np
import matplotlib.pyplot as plt

from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_utils import overlay_timesteps

if __name__ == "__main__":
    np.random.seed(2)

    # generate initial conditions
    ts = 0.05
    xh0 = np.random.uniform(-10, 10, (4, 1))
    xh0[[1,3]] = 0
    xr0 = np.random.uniform(-10, 10, (4, 1))
    xr0[[1,3]] = 0
    goals = np.random.uniform(-10, 10, (4, 3))
    goals[[1,3]] = 0
    r_goal = goals[:,[1]] 

    # create human and robot objects
    W = np.diag([0.0, 0.0, 0.0, 0.0])
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    robot.set_goals(goals)

    # run simulation
    arr_size = 100
    all_h_beliefs = np.zeros((0, goals.shape[1]))
    xr_traj = xr0

    fig, axes = plt.subplots(1, 2)
    axes = axes.flatten()
    ax = axes[0]
    ax.set_aspect('equal', adjustable='box')
    h_belief_ax = axes[1]
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

    for idx in range(arr_size):
        # plotting
        ax.cla()
        # plot trajectory trail so far
        overlay_timesteps(ax, [], xr_traj, n_steps=idx)

        ax.scatter(goals[0], goals[2], c=goal_colors, s=100)
        ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        h_belief_ax.clear()
        # import ipdb; ipdb.set_trace()
        h_belief_ax.plot(all_h_beliefs[:,0], label="P(g0)", c=goal_colors[0])
        h_belief_ax.plot(all_h_beliefs[:,1], label="P(g1)", c=goal_colors[1])
        h_belief_ax.plot(all_h_beliefs[:,2], label="P(g2)", c=goal_colors[2])
        h_belief_ax.set_xlabel("h belief of r")
        h_belief_ax.legend()
        plt.pause(0.001)

        # select robot action
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
        print(f"{idx}: {np.linalg.norm(ur)}")
        # if idx >= 10:
        #     import ipdb; ipdb.set_trace()

        # belief update
        belief.update_belief(robot.x, ur)

        # take action
        xr0 = robot.step(ur)

        if all_h_beliefs is not None:
            all_h_beliefs = np.vstack((all_h_beliefs, belief.belief))
        # xr_traj[:,[idx+1]] = xr0
        xr_traj = np.hstack((xr_traj, xr0))
    plt.show()
