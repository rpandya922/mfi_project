import numpy as np
import matplotlib.pyplot as plt

from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman  
from robot import Robot
from intention_utils import overlay_timesteps
from cbp_model import CBPEstimator
from generate_path_loop_multiple_obs import generate_trajectory_belief

def run_simulation():
    # generate initial conditions
    np.random.seed(287)
    ts = 0.1
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
    # simulate for T seconds
    N = int(10 / ts)
    traj_horizon = int(2 / ts)

    h_beliefs = h_belief.belief
    r_beliefs = r_belief.belief
    r_beliefs_nominal = r_belief_nominal.belief
    r_belief_likelihoods = []
    h_goal_idxs = []
    r_goal_idxs = []

    # figure for plotting
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    axes = np.array(axes).flatten()
    ax = axes[0]
    # make ax equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    h_belief_ax = axes[1]
    r_belief_ax = axes[2]
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
    h_goal_ax = axes[3]
    r_beta_ax = axes[4]
    r_likelihoods_ax = axes[5]

    for idx in range(N):
        # get human and robot controls
        uh = human.dynamics.get_goal_control(human.x, human.get_goal())
        # generate safe trajectory for robot
        safety, ur_traj, obs_loc = generate_trajectory_belief(robot, human, r_belief, traj_horizon, goals, plot=False)
        # if robot is sufficiently close to its goal, just move straight to the goal without considering human
        if np.linalg.norm(robot.x[[0,2]] - robot.goal[[0,2]]) < 1.5:
            ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
        else:
            ur = ur_traj[:,[0]]

        # update human's belief
        human.update_belief(robot.x, ur)

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
            h_belief_score = p[np.argmax(r_belief_prior)]
            goal_change_score = -0.01*np.linalg.norm(goals[:,[goal_idx]] - robot.goal)
            goal_change_score = 0
            goal_scores.append(h_belief_score + goal_change_score)
        goal_idx = np.argmax(goal_scores)
        robot.goal = goals[:,[goal_idx]]
        
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
        # r_beliefs_beta = np.dstack((r_beliefs_beta, r_belief_beta.belief))
        r_belief_likelihoods.append(likelihoods)
        # save human's actual intended goal
        h_goal_idxs.append(np.argmin(np.linalg.norm(human.goal - goals, axis=0)))
        # save robot's actual intended goal
        r_goal_idxs.append(np.argmin(np.linalg.norm(robot.goal - goals, axis=0)))

        # plot
        ax.clear()
        overlay_timesteps(ax, xh_traj, xr_traj, n_steps=idx)
        ax.scatter(xh0[0], xh0[2], c="blue")
        ax.scatter(xr0[0], xr0[2], c="red")
        ax.scatter(obs_loc[0], obs_loc[2], c="yellow")
        ax.scatter(goals[0], goals[2], c=goal_colors)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        h_belief_ax.clear()
        h_belief_ax.plot(h_beliefs[:,0], label="P(g0)", c=goal_colors[0])
        h_belief_ax.plot(h_beliefs[:,1], label="P(g1)", c=goal_colors[1])
        h_belief_ax.plot(h_beliefs[:,2], label="P(g2)", c=goal_colors[2])
        h_belief_ax.set_xlabel("h belief of r")
        h_belief_ax.legend()

        r_belief_ax.clear()
        r_belief_ax.plot(r_beliefs[:,0], label="P(g0)", c=goal_colors[0])
        r_belief_ax.plot(r_beliefs[:,1], label="P(g1)", c=goal_colors[1])
        r_belief_ax.plot(r_beliefs[:,2], label="P(g2)", c=goal_colors[2])
        # plot nomninal belief with dashed lines
        r_belief_ax.plot(r_beliefs_nominal[:,0], c=goal_colors[0], linestyle="--")
        r_belief_ax.plot(r_beliefs_nominal[:,1], c=goal_colors[1], linestyle="--")
        r_belief_ax.plot(r_beliefs_nominal[:,2], c=goal_colors[2], linestyle="--")
        r_belief_ax.set_xlabel("r belief of h")
        r_belief_ax.legend()

        h_goal_ax.clear()
        h_goal_ax.plot(h_goal_idxs, c="blue", label="h goal")
        h_goal_ax.plot(r_goal_idxs, c="red", label="r goal")
        h_goal_ax.legend()

        r_beta_ax.clear()
        # for now, plotting marginalized belief
        # for beta_idx in range(r_belief_beta.betas.shape[0]):
        #     r_beta_ax.plot(r_beliefs_beta[:,beta_idx,:].sum(axis=0), label=f"b={r_belief_beta.betas[beta_idx]}")
        # r_beta_ax.legend()

        r_likelihoods_ax.clear()
        l = np.array(r_belief_likelihoods)
        for theta_idx in range(r_belief_nominal.thetas.shape[1]):
            r_likelihoods_ax.plot(l[:,theta_idx], label=f"theta={theta_idx}", c=goal_colors[theta_idx])
        r_likelihoods_ax.legend()

        plt.pause(0.01)
    plt.show()

if __name__ == "__main__":
    run_simulation()