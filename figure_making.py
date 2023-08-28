import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_utils import overlay_timesteps
from cbp_model import CBPEstimator

def chattering_goals(reward="kl"):
    # np.random.seed(5) # reasonable seed
    np.random.seed(6)

    ts = 0.1
    # simulate for T seconds
    N = int(10 / ts)
    traj_horizon = int(2 / ts)

    xh0 = np.random.uniform(-10, 10, (4, 1))
    xh0[[1,3]] = 0
    xr0 = np.random.uniform(-10, 10, (4, 1))
    xr0[[1,3]] = 0
    goals = np.random.uniform(-10, 10, (4, 3))
    goals[[1,3]] = 0
    r_goal = goals[:,[2]] # this is arbitrary since it'll be changed in simulations later anyways

    # create human and robot objects
    # W = np.diag([0.0, 0.7, 0.0, 0.7])
    W = np.diag([0.0, 0.0, 0.0, 0.0])
    h_dynamics = DIDynamics(ts=ts, W=W)
    r_dynamics = DIDynamics(ts=ts)

    h_belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.005)
    human = BayesHuman(xh0, h_dynamics, goals, h_belief, gamma=5)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    r_belief = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.005)
    # r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.005)

    # trajectory saving
    xh_traj = xh0
    xr_traj = xr0

    h_beliefs = h_belief.belief
    r_beliefs = r_belief.belief
    r_beliefs_nominal = r_belief_nominal.belief
    h_goal_idxs = []
    r_goal_idxs = []

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    axes = axes.flatten()
    ax = axes[0]
    h_goal_ax = axes[1]
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

    for idx in range(N):
        # get human and robot controls
        uh = human.get_u(robot.x)

        # update robot's nominal belief
        r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, return_likelihood=True)
        r_belief_prior = r_belief_nominal.belief

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
            if reward == "kl":
                h_belief_score = entropy(posts[goal_idx], r_belief_prior)
            elif reward == "max":
                h_belief_score = p[np.argmax(r_belief_prior)]
            # goal_change_score = -0.01*np.linalg.norm(goals[:,[goal_idx]] - robot.goal)
            goal_change_score = 0
            goal_scores.append(h_belief_score + goal_change_score)
        goal_idx = np.argmax(goal_scores)
        robot.goal = goals[:,[goal_idx]]

        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # update human's belief
        if np.linalg.norm(ur) > 1e-3:
            human.update_belief(robot.x, ur)
            # set min belief to 0.01
            human.belief.belief = np.maximum(human.belief.belief, 0.01)

        # update robot's belief
        r_belief_post = posts[goal_idx]
        r_belief.belief = r_belief_post

        # step dynamics forward
        xh0 = human.step(uh)
        xr0 = robot.step(ur)

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))
        h_beliefs = np.vstack((h_beliefs, h_belief.belief))
        r_beliefs = np.vstack((r_beliefs, r_belief.belief))
        r_beliefs_nominal = np.vstack((r_beliefs_nominal, r_belief_nominal.belief))
        # save human's actual intended goal
        h_goal_idxs.append(np.argmin(np.linalg.norm(human.goal - goals, axis=0)))
        # save robot's actual intended goal
        r_goal_idxs.append(np.argmin(np.linalg.norm(robot.goal - goals, axis=0)))

        h_goal_reached = False
        if np.linalg.norm(human.x[[0,2]] - human.goal[[0,2]]) < 0.5:
            h_goal_reached = True
        r_goal_reached = False
        if np.linalg.norm(robot.x[[0,2]] - robot.goal[[0,2]]) < 0.5:
            r_goal_reached = True
    
        # plot
        ax.clear()
        overlay_timesteps(ax, xh_traj, xr_traj, n_steps=idx)
        ax.scatter(xh0[0], xh0[2], c="blue")
        ax.scatter(xr0[0], xr0[2], c="red")
        ax.scatter(goals[0], goals[2], c=goal_colors)
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 10)

        h_goal_ax.clear()
        h_goal_ax.plot(h_goal_idxs, c="blue", label="h goal")
        h_goal_ax.plot(r_goal_idxs, c="red", label="r goal")
        h_goal_ax.legend()

        # for goal_idx in range(r_beliefs.shape[1]):
        #     h_goal_ax.plot(r_beliefs[:,goal_idx], label=f"P(g{goal_idx})", c=goal_colors[goal_idx])
        # # plot nomninal belief with dashed lines
        # for goal_idx in range(r_beliefs_nominal.shape[1]):
        #     h_goal_ax.plot(r_beliefs_nominal[:,goal_idx], c=goal_colors[goal_idx], linestyle="--")

        # plt.pause(0.01)

        # if h_goal_reached and r_goal_reached:
        #     # stop early if both goals are reached
        #     break

    plt.show()
        
def influence_objective(mode="with_robot"):
    np.random.seed(0)

    ts = 0.1
    # simulate for T seconds
    N = int(5 / ts)

    xh0 = np.random.uniform(-10, 10, (4, 1))
    xh0[[1,3]] = 0
    xr0 = np.random.uniform(-10, 10, (4, 1))
    xr0[[1,3]] = 0
    goals = np.random.uniform(-10, 10, (4, 3))
    goals[[1,3]] = 0
    r_goal = goals[:,[2]] # this is arbitrary since it'll be changed in simulations later anyways

    # create human and robot objects
    # W = np.diag([0.0, 0.7, 0.0, 0.7])
    W = np.diag([0.0, 0.0, 0.0, 0.0])
    h_dynamics = DIDynamics(ts=ts, W=W)
    r_dynamics = DIDynamics(ts=ts)

    h_belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.0005)
    human = BayesHuman(xh0, h_dynamics, goals, h_belief, gamma=5)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    r_belief = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)

    # trajectory saving
    xh_traj = xh0
    xr_traj = xr0

    h_beliefs = h_belief.belief
    r_beliefs = r_belief.belief
    r_beliefs_nominal = r_belief_nominal.belief
    h_goal_idxs = []
    r_goal_idxs = []

    fig, ax = plt.subplots(figsize=(7,5))
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

    for idx in range(N):
        # get human and robot controls
        if mode == "with_robot":
            uh = human.get_u(robot.x)
        else:
            uh = human.get_u(-100*np.ones((4,1)))

        # update robot's nominal belief
        r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, return_likelihood=True)
        r_belief_prior = r_belief_nominal.belief

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
            theta_hat = goals[:,[np.argmax(p)]]
            xr_dist = np.linalg.norm(robot.x - goals[:,[goal_idx]])
            xh_dist = np.linalg.norm(human.x - theta_hat)
            h_belief_score = xr_dist + xh_dist
            goal_change_score = 0
            goal_scores.append(h_belief_score + goal_change_score)
        goal_idx = np.argmin(goal_scores)
        robot.goal = goals[:,[goal_idx]]


        if mode == "with_robot":
            ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

            # update human's belief
            if np.linalg.norm(ur) > 1e-3:
                human.update_belief(robot.x, ur)
                # set min belief to 0.01
                human.belief.belief = np.maximum(human.belief.belief, 0.01)
        else:
            ur = np.zeros((2,1))

        # update robot's belief
        r_belief_post = posts[goal_idx]
        r_belief.belief = r_belief_post

        # step dynamics forward
        xh0 = human.step(uh)
        xr0 = robot.step(ur)

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))
        h_beliefs = np.vstack((h_beliefs, h_belief.belief))
        r_beliefs = np.vstack((r_beliefs, r_belief.belief))
        r_beliefs_nominal = np.vstack((r_beliefs_nominal, r_belief_nominal.belief))
        # save human's actual intended goal
        h_goal_idxs.append(np.argmin(np.linalg.norm(human.goal - goals, axis=0)))
        # save robot's actual intended goal
        r_goal_idxs.append(np.argmin(np.linalg.norm(robot.goal - goals, axis=0)))

        h_goal_reached = False
        if np.linalg.norm(human.x[[0,2]] - human.goal[[0,2]]) < 0.5:
            h_goal_reached = True
        r_goal_reached = False
        if np.linalg.norm(robot.x[[0,2]] - robot.goal[[0,2]]) < 0.5:
            r_goal_reached = True
    
        # plot
        ax.clear()
        if mode == "with_robot":
            overlay_timesteps(ax, xh_traj, xr_traj, n_steps=idx)
            ax.scatter(xr0[0], xr0[2], c="red")
        else:
            overlay_timesteps(ax, xh_traj, [], n_steps=idx)
        ax.scatter(xh0[0], xh0[2], c="blue")
        ax.scatter(goals[0], goals[2], c=goal_colors)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal", "box")

        plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    # chattering_goals("max")
    # chattering_goals("kl")

    # influence_objective("no_robot")
    influence_objective("with_robot")
