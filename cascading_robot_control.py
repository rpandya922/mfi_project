import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
from tqdm import tqdm
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from intention_predictor import create_model
from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman  
from robot import Robot
from intention_utils import overlay_timesteps
from cbp_model import CBPEstimator, BetaBayesEstimator
from generate_path_loop_multiple_obs import generate_trajectory_belief
from generate_path_nn_full_loop import generate_trajectories
from cbp_nn import compute_post_nn

class UncertainHuman(BayesHuman):
    def __init__(self, x0, dynamics, goals, belief : BayesEstimator, gamma=1, mode="waiting"):
        super(UncertainHuman, self).__init__(x0, dynamics, goals, belief, gamma=gamma)

        self.mode = mode
    
    def get_u(self, robot_x):
        if self.mode == "waiting":
            
            # switch to moving mode if belief on robot is certain enough
            if self.belief.belief.max() > 0.8:
                # self.mode = "moving"
                return super(UncertainHuman, self).get_u(robot_x)
            
            return np.zeros((2,1))
        elif self.mode == "moving":
            return super(UncertainHuman, self).get_u(robot_x)
        elif self.mode == "stubborn":
            # always move towards goal
            self.goal = self.goals[:,[np.linalg.norm(self.goals - self.x, axis=0).argmin()]]
            return self.dynamics.get_goal_control(self.x, self.goal)

def run_simulation(robot_type="cbp", human_type="moving", plot=True):
    # robot_type should be "cbp", "baseline", "baseline_belief" or "cbp_nn"
    # generate initial conditions
    # np.random.seed(287)
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
    W = np.diag([0.0, 0.7, 0.0, 0.7])
    # W = np.diag([0.0, 0.0, 0.0, 0.0])
    h_dynamics = DIDynamics(ts=ts, W=W)
    r_dynamics = DIDynamics(ts=ts)

    h_belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.0005)
    human = UncertainHuman(xh0, h_dynamics, goals, h_belief, gamma=5, mode=human_type)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    r_belief = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    # r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0001)
    # r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3], dynamics=h_dynamics) # working with [5e-5, 5e-3]
    r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[5e-6, 5e-5, 5e-3], dynamics=h_dynamics) # working with [5e-5, 5e-3]

    if robot_type == "cbp_nn":
        stats_file = "./data/prob_pred/bayes_prob_branching_processed_feats_ts01_stats.pkl" # working ts=0.1
        model_path = "./data/models/prob_pred_intention_predictor_bayes_20230818-174117.pt" # working ts=0.1
        k_hist = 5
        model = create_model(horizon_len=traj_horizon, hidden_size=128, num_layers=2, hist_feats=8, plan_feats=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with open(stats_file, "rb") as f:
            stats = pickle.load(f)

    # trajectory saving
    xh_traj = xh0
    xr_traj = xr0

    h_beliefs = h_belief.belief
    r_beliefs = r_belief.belief
    r_beliefs_nominal = r_belief_nominal.belief
    r_beliefs_beta = r_belief_beta.belief
    r_belief_likelihoods = []
    h_goal_idxs = []
    r_goal_idxs = []
    is_robot_waiting = []
    is_human_waiting = []

    if plot:
        # figure for plotting
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
        axes = np.array(axes).flatten()
        ax = axes[0]
        # make ax equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        h_belief_ax = axes[1]
        r_belief_ax = axes[2]
        # goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
        goal_colors = ["#3A637B", "#C4A46B", "#FF5A00", "green", "purple"]
        h_goal_ax = axes[3]
        r_beta_ax = axes[4]
        r_likelihoods_ax = axes[5]
    
    robot_wait_time = 5
    h_goal_reached = False
    r_goal_reached = False
    # robot_wait_time = 0
    for idx in range(N):
        # get human and robot controls
        uh = human.get_u(robot.x)
        if np.linalg.norm(uh) < 1e-5:
            is_human_waiting.append(True)
        else:
            is_human_waiting.append(False)

        # update robot's nominal belief
        r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, return_likelihood=True)
        r_belief_prior = r_belief_nominal.belief
        r_belief_beta.belief = r_belief_beta.update_belief(human.x, uh)

        # check if the most likely beta is below a threshold for all theta. if it is, the human is waiting around
        human_waiting = False
        if (r_belief_beta.betas[np.argmax(r_belief_beta.belief, axis=1)] < 5e-3).all():
            human_waiting = True

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

        # update human's belief
        if np.linalg.norm(ur) > 1e-3:
            human.update_belief(robot.x, ur)
            # set min belief to 0.01
            human.belief.belief = np.maximum(human.belief.belief, 0.01)

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
            dists[r_belief_nominal.belief.argmax()] = np.inf
            goal_idx = dists.argmin()

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
        r_beliefs_beta = np.dstack((r_beliefs_beta, r_belief_beta.belief))
        r_belief_likelihoods.append(likelihoods)
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
        if h_goal_reached and r_goal_reached:
            # stop early if both goals are reached
            break

        if plot:
            # plot
            ax.clear()
            overlay_timesteps(ax, xh_traj, xr_traj, n_steps=idx)
            ax.scatter(xh0[0], xh0[2], c="blue")
            ax.scatter(xr0[0], xr0[2], c="red")
            if obs_loc is not None:
                ax.scatter(obs_loc[0], obs_loc[2], c="yellow")
            ax.scatter(goals[0], goals[2], c=goal_colors)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

            h_belief_ax.clear()
            for goal_idx in range(h_beliefs.shape[1]):
                h_belief_ax.plot(h_beliefs[:,goal_idx], label=f"P(g{goal_idx})", c=goal_colors[goal_idx])
            h_belief_ax.set_xlabel("h belief of r")
            h_belief_ax.legend()

            r_belief_ax.clear()
            for goal_idx in range(r_beliefs.shape[1]):
                r_belief_ax.plot(r_beliefs[:,goal_idx], label=f"P(g{goal_idx})", c=goal_colors[goal_idx])
            # plot nomninal belief with dashed lines
            for goal_idx in range(r_beliefs_nominal.shape[1]):
                r_belief_ax.plot(r_beliefs_nominal[:,goal_idx], c=goal_colors[goal_idx], linestyle="--")
            r_belief_ax.set_xlabel("r belief of h")
            r_belief_ax.legend()

            h_goal_ax.clear()
            h_goal_ax.plot(h_goal_idxs, c="blue", label="h goal")
            h_goal_ax.plot(r_goal_idxs, c="red", label="r goal")
            h_goal_ax.legend()

            r_beta_ax.clear()
            # for now, plotting marginalized belief
            for beta_idx in range(r_belief_beta.betas.shape[0]):
                r_beta_ax.plot(r_beliefs_beta[:,beta_idx,:].sum(axis=0), label=f"b={r_belief_beta.betas[beta_idx]}")
            r_beta_ax.legend()

            r_likelihoods_ax.clear()
            l = np.array(r_belief_likelihoods)
            for theta_idx in range(r_belief_nominal.thetas.shape[1]):
                r_likelihoods_ax.plot(l[:,theta_idx], label=f"theta={theta_idx}", c=goal_colors[theta_idx])
            # for theta_idx in range(r_belief_beta.thetas.shape[1]):
            #     r_likelihoods_ax.plot(r_beliefs_beta[theta_idx,:,:].sum(axis=0), label=f"theta={theta_idx}", c=goal_colors[theta_idx])
            r_likelihoods_ax.legend()

            plt.pause(0.01)
    if plot:
        plt.show()

    data = {"xh_traj": xh_traj, "xr_traj": xr_traj, "h_beliefs": h_beliefs, "r_beliefs": r_beliefs, "r_beliefs_nominal": r_beliefs_nominal, "r_beliefs_beta": r_beliefs_beta, "r_belief_likelihoods": r_belief_likelihoods, "h_goal_idxs": h_goal_idxs, "r_goal_idxs": r_goal_idxs, "goals": goals, "robot_type": robot_type, "human_type": human_type, "is_robot_waiting": is_robot_waiting, "is_human_waiting": is_human_waiting}

    return data

def run_simulation_nn(robot_type="cbp_nn", human_type="moving", plot=True):
    # robot_type should be "cbp_nn"
    # generate initial conditions
    # np.random.seed(287)
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
    W = np.diag([0.0, 0.7, 0.0, 0.7])
    # W = np.diag([0.0, 0.0, 0.0, 0.0])
    h_dynamics = DIDynamics(ts=ts, W=W)
    r_dynamics = DIDynamics(ts=ts)

    h_belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.0005)
    human = UncertainHuman(xh0, h_dynamics, goals, h_belief, gamma=5, mode=human_type)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    r_belief = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    # r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0001)
    # r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3], dynamics=h_dynamics) # working with [5e-5, 5e-3]
    r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[5e-6, 5e-5, 5e-3], dynamics=h_dynamics) # working with [5e-5, 5e-3]

    if robot_type == "cbp_nn":
        stats_file = "./data/models/bayes_prob_branching_processed_feats_ts01_safe_stats.pkl" # working ts=0.1
        # model_path = "./data/models/prob_pred_intention_predictor_bayes_20230818-174117.pt" # working ts=0.1
        model_path = "./data/models/prob_pred_intention_predictor_bayes_20230902-112136.pt" # working ts=0.1
        k_hist = 5
        model = create_model(horizon_len=traj_horizon, hidden_size=128, num_layers=2, hist_feats=21, plan_feats=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with open(stats_file, "rb") as f:
            stats = pickle.load(f)
    else:
        raise ValueError("Invalid robot type")

    # trajectory saving
    xh_traj = xh0
    xr_traj = xr0

    h_beliefs = h_belief.belief
    r_beliefs = r_belief.belief
    r_beliefs_nominal = r_belief_nominal.belief
    r_beliefs_beta = r_belief_beta.belief
    r_belief_likelihoods = []
    h_goal_idxs = []
    r_goal_idxs = []
    is_robot_waiting = []
    is_human_waiting = []

    if plot:
        # figure for plotting
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
        axes = np.array(axes).flatten()
        ax = axes[0]
        # make ax equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        h_belief_ax = axes[1]
        r_belief_ax = axes[2]
        goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
        # goal_colors = ["#3A637B", "#C4A46B", "#FF5A00", "green", "purple"]
        h_goal_ax = axes[3]
        r_beta_ax = axes[4]
        r_likelihoods_ax = axes[5]
    
    # TODO: change sim to continue adding goals 
    robot_wait_time = 5
    h_goal_reached = False
    r_goal_reached = False
    # robot_wait_time = 0
    for idx in range(N):
        # get human and robot controls
        uh = human.get_u(robot.x)
        if np.linalg.norm(uh) < 1e-5:
            is_human_waiting.append(True)
        else:
            is_human_waiting.append(False)

        # update robot's nominal belief
        r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, return_likelihood=True)
        r_belief_prior = r_belief_nominal.belief
        r_belief_beta.belief = r_belief_beta.update_belief(human.x, uh)

        # check if the most likely beta is below a threshold for all theta. if it is, the human is waiting around
        human_waiting = False
        if (r_belief_beta.betas[np.argmax(r_belief_beta.belief, axis=1)] < 5e-3).all():
            human_waiting = True

        # loop through goals and compute belief update for each potential robot goal
        posts = []
        ur_trajs = []
        for goal_idx in range(goals.shape[1]):
            goal = goals[:,[goal_idx]]

            # get xr and xh hist for NN input
            if idx < k_hist:
                # get zero-padded history of both agents
                xh_hist = np.hstack((np.zeros((human.dynamics.n, k_hist - idx)), xh_traj[:, 0:idx]))
                xr_hist = np.hstack((np.zeros((robot.dynamics.n, k_hist - idx)), xr_traj[:, 0:idx]))
            else:
                xh_hist = xh_traj[:, idx - k_hist:idx]
                xr_hist = xr_traj[:, idx - k_hist:idx]
            
            # generate safe trajectory for robot given this goal
            safety, ur_traj, distance, new_belief, obs_loc = generate_trajectories(robot, human, r_belief_nominal.belief, traj_horizon, goals, goal, xh_hist, xr_hist, model, stats_file=None, stats=stats, verbose=False, plot=False)

            posts.append(new_belief)
            ur_trajs.append(ur_traj)

        # pick goal with highest probability on human's most likely goal
        goal_scores = []
        for goal_idx in range(goals.shape[1]):
            p = posts[goal_idx]
            h_belief_score = p[np.argmax(r_belief_prior)]
            goal_change_score = -0.01*np.linalg.norm(goals[:,[goal_idx]] - robot.goal)
            # goal_change_score = 0
            goal_scores.append(h_belief_score + goal_change_score)
        goal_idx = np.argmax(goal_scores)
        ur_traj = ur_trajs[goal_idx]

        if not human_waiting and robot_wait_time > 0 and robot_type == "cbp_nn":
            ur = np.zeros((2,1))
            obs_loc = None
            robot_wait_time -= 1
            is_robot_waiting.append(True)
        else:
            robot_wait_time = 0 # make sure robot commits to waiting only at the start
            # if robot is sufficiently close to its goal, just move straight to the goal without considering human
            if (np.linalg.norm(robot.x[[0,2]] - robot.goal[[0,2]]) < 1.5) or h_goal_reached:
                ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
            else:
                ur = ur_traj[:,[0]]
            is_robot_waiting.append(False)

        # update human's belief
        if np.linalg.norm(ur) > 1e-3:
            human.update_belief(robot.x, ur)
            # set min belief to 0.01
            human.belief.belief = np.maximum(human.belief.belief, 0.01)

        if robot_type == "baseline":
            goal_idx = np.linalg.norm(robot.x - goals, axis=0).argmin()
        elif robot_type == "baseline_belief":
            # pick the closest goal that the human is not moving towards
            dists = np.linalg.norm(robot.x - goals, axis=0)
            dists[r_belief_nominal.belief.argmax()] = np.inf
            goal_idx = dists.argmin()

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
        r_beliefs_beta = np.dstack((r_beliefs_beta, r_belief_beta.belief))
        r_belief_likelihoods.append(likelihoods)
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
        if h_goal_reached and r_goal_reached:
            # stop early if both goals are reached
            break

        if plot:
            # plot
            ax.clear()
            overlay_timesteps(ax, xh_traj, xr_traj, n_steps=idx)
            ax.scatter(xh0[0], xh0[2], c="blue")
            ax.scatter(xr0[0], xr0[2], c="red")
            if obs_loc is not None:
                ax.scatter(obs_loc[0], obs_loc[2], c="yellow")
            ax.scatter(goals[0], goals[2], c=goal_colors)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

            h_belief_ax.clear()
            for goal_idx in range(h_beliefs.shape[1]):
                h_belief_ax.plot(h_beliefs[:,goal_idx], label=f"P(g{goal_idx})", c=goal_colors[goal_idx])
            h_belief_ax.set_xlabel("h belief of r")
            h_belief_ax.legend()

            r_belief_ax.clear()
            for goal_idx in range(r_beliefs.shape[1]):
                r_belief_ax.plot(r_beliefs[:,goal_idx], label=f"P(g{goal_idx})", c=goal_colors[goal_idx])
            # plot nomninal belief with dashed lines
            for goal_idx in range(r_beliefs_nominal.shape[1]):
                r_belief_ax.plot(r_beliefs_nominal[:,goal_idx], c=goal_colors[goal_idx], linestyle="--")
            r_belief_ax.set_xlabel("r belief of h")
            r_belief_ax.legend()

            h_goal_ax.clear()
            h_goal_ax.plot(h_goal_idxs, c="blue", label="h goal")
            h_goal_ax.plot(r_goal_idxs, c="red", label="r goal")
            h_goal_ax.legend()

            r_beta_ax.clear()
            # for now, plotting marginalized belief
            for beta_idx in range(r_belief_beta.betas.shape[0]):
                r_beta_ax.plot(r_beliefs_beta[:,beta_idx,:].sum(axis=0), label=f"b={r_belief_beta.betas[beta_idx]}")
            r_beta_ax.legend()

            r_likelihoods_ax.clear()
            l = np.array(r_belief_likelihoods)
            for theta_idx in range(r_belief_nominal.thetas.shape[1]):
                r_likelihoods_ax.plot(l[:,theta_idx], label=f"theta={theta_idx}", c=goal_colors[theta_idx])
            # for theta_idx in range(r_belief_beta.thetas.shape[1]):
            #     r_likelihoods_ax.plot(r_beliefs_beta[theta_idx,:,:].sum(axis=0), label=f"theta={theta_idx}", c=goal_colors[theta_idx])
            r_likelihoods_ax.legend()

            plt.pause(0.01)
    if plot:
        plt.show()

    data = {"xh_traj": xh_traj, "xr_traj": xr_traj, "h_beliefs": h_beliefs, "r_beliefs": r_beliefs, "r_beliefs_nominal": r_beliefs_nominal, "r_beliefs_beta": r_beliefs_beta, "r_belief_likelihoods": r_belief_likelihoods, "h_goal_idxs": h_goal_idxs, "r_goal_idxs": r_goal_idxs, "goals": goals, "robot_type": robot_type, "human_type": human_type, "is_robot_waiting": is_robot_waiting, "is_human_waiting": is_human_waiting}

    return data

def run_simulations(filepath="./data/sim_stats.pkl", n_traj=10):
    # robot_types = ["cbp", "baseline", "baseline_belief", "cbp_nn"]
    robot_types = ["cbp_nn"]
    all_stats = {robot_type: [] for robot_type in robot_types}
    for robot_type in robot_types:
        # TODO: standardize initial conditions better (since traj len may change, this way doesn't work)
        np.random.seed(0)
        for i in tqdm(range(n_traj)):
            if robot_type == "cbp_nn":
                data = run_simulation_nn(robot_type=robot_type, human_type="moving", plot=False)
            else:
                data = run_simulation(robot_type=robot_type, human_type="moving", plot=False)
            all_stats[robot_type].append(data)
    # save all_stats
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(all_stats, f)

def influence_obj(human, robot, goals, belief_prior, posts):
    goal_scores = []
    for goal_idx in range(goals.shape[1]):
        r_goal = goals[:,[goal_idx]]
        post_goal = posts[goal_idx]
        h_goal_idx = np.argmax(post_goal)
        h_goal = goals[:,[h_goal_idx]]
        score = np.linalg.norm(human.x - h_goal) + np.linalg.norm(robot.x - r_goal)
        goal_scores.append(score)
    return np.argmin(goal_scores)

def courtesy_obj(human, robot, goals, belief_prior, posts):
    goal_scores = []
    for goal_idx in range(goals.shape[1]):
        p = posts[goal_idx]
        h_belief_score = p[np.argmax(belief_prior)]
        goal_change_score = -0.02*np.linalg.norm(goals[:,[goal_idx]] - robot.goal)
        goal_scores.append(h_belief_score + goal_change_score)
    return np.argmax(goal_scores)

def run_full_game(robot_type="cbp", human_type="moving", plot=True):
    # robot_type should be "cbp", "baseline", "baseline_belief" or "cbp_nn"
    # generate initial conditions
    # np.random.seed(287)
    ts = 0.1
    # simulate for T seconds
    N = int(30 / ts)
    traj_horizon = int(2 / ts)
    n_goals = 3

    xh0 = np.random.uniform(-10, 10, (4, 1))
    xh0[[1,3]] = 0
    xr0 = np.random.uniform(-10, 10, (4, 1))
    xr0[[1,3]] = 0
    goals = np.random.uniform(-10, 10, (4, n_goals))
    goals[[1,3]] = 0
    r_goal = goals[:,[2]] # this is arbitrary since it'll be changed in simulations later anyways

    # create human and robot objects
    W = np.diag([0.0, 0.7, 0.0, 0.7])
    # W = np.diag([0.0, 0.0, 0.0, 0.0])
    h_dynamics = DIDynamics(ts=ts, W=W)
    r_dynamics = DIDynamics(ts=ts)

    h_belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.0003)
    human = UncertainHuman(xh0, h_dynamics, goals, h_belief, gamma=5, mode=human_type)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    r_belief = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    # r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0001)
    # r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3], dynamics=h_dynamics) # working with [5e-5, 5e-3]
    r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[5e-6, 5e-5, 5e-3], dynamics=h_dynamics) # working with [5e-5, 5e-3]

    if robot_type == "cbp_nn":
        stats_file = "./data/models/bayes_prob_branching_processed_feats_ts01_safe_stats.pkl" # working ts=0.1
        # model_path = "./data/models/prob_pred_intention_predictor_bayes_20230818-174117.pt" # working ts=0.1
        model_path = "./data/models/prob_pred_intention_predictor_bayes_20230902-112136.pt" # working ts=0.1
        k_hist = 5
        model = create_model(horizon_len=traj_horizon, hidden_size=128, num_layers=2, hist_feats=21, plan_feats=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with open(stats_file, "rb") as f:
            stats = pickle.load(f)
    # else:
    #     raise ValueError("Invalid robot type")

    # trajectory saving
    xh_traj = xh0
    xr_traj = xr0

    h_beliefs = h_belief.belief
    r_beliefs = r_belief.belief
    r_beliefs_nominal = r_belief_nominal.belief
    r_beliefs_beta = r_belief_beta.belief
    r_belief_likelihoods = []
    h_goal_idxs = []
    r_goal_idxs = []
    is_robot_waiting = []
    is_human_waiting = []
    all_goals = goals

    if plot:
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
    
    robot_wait_time = 5
    h_goal_reached = False
    r_goal_reached = False
    # robot_wait_time = 0
    both_at_goal = False
    team_score = 0
    for idx in range(N):

        if both_at_goal:
            # resample goals
            # find goal human is at
            h_goal_idx = np.argmin(np.linalg.norm(xh0 - goals, axis=0))
            r_goal_idx = np.argmin(np.linalg.norm(xr0 - goals, axis=0))
            
            new_goals = np.random.uniform(size=(4, 2))*20 - 10
            new_goals[[1,3],:] = np.zeros((2, 2))
            goals[:,[h_goal_idx, r_goal_idx]] = new_goals

            # reset beliefs
            r_belief.belief = np.ones(n_goals) / n_goals
            r_belief.thetas = goals
            r_belief_nominal.belief = np.ones(n_goals) / n_goals
            r_belief_nominal.thetas = goals
            r_belief_beta.belief = np.ones((n_goals, len(r_belief_beta.betas))) / (n_goals*len(r_belief_beta.betas))
            r_belief_beta.thetas = goals
            human.belief.belief = np.ones(n_goals) / n_goals
            human.belief.thetas = goals

            human.goals = goals
            robot.goals = goals

            human.get_goal()
            robot.goal = goals[:,[np.linalg.norm(robot.x - goals, axis=0).argmin()]]

            # increase score
            team_score += 2

        # get human and robot controls
        uh = human.get_u(robot.x)
        if np.linalg.norm(uh) < 1e-5:
            is_human_waiting.append(True)
        else:
            is_human_waiting.append(False)

        # update robot's nominal belief
        r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, return_likelihood=True)
        r_belief_prior = r_belief_nominal.belief
        r_belief_beta.belief = r_belief_beta.update_belief(human.x, uh)

        # check if the most likely beta is below a threshold for all theta. if it is, the human is waiting around
        human_waiting = False
        if (r_belief_beta.betas[np.argmax(r_belief_beta.belief, axis=1)] < 5e-3).all():
            human_waiting = True

        if robot_type != "cbp_nn":
            if np.linalg.norm(robot.x[[0,2]] - robot.goal[[0,2]]) >= 1:
                safety, ur_traj, obs_loc = generate_trajectory_belief(robot, human, r_belief, traj_horizon, goals, plot=False)
                ur = ur_traj[:,[0]]
            else:
                ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

            xh_next = human.dynamics.A @ human.x + human.dynamics.B @ uh
            # loop through goals and compute belief update for each potential robot goal
            posts = []
            for goal_idx in range(goals.shape[1]):
                goal = goals[:,[goal_idx]]
                # compute CBP belief update
                r_belief_post = r_belief.weight_by_score(r_belief_prior, goal, xh_next, beta=0.5)
                posts.append(r_belief_post)
        else:
            # loop through goals and compute belief update for each potential robot goal
            posts = []
            ur_trajs = []
            for goal_idx in range(goals.shape[1]):
                goal = goals[:,[goal_idx]]

                # get xr and xh hist for NN input
                if idx < k_hist:
                    # get zero-padded history of both agents
                    xh_hist = np.hstack((np.zeros((human.dynamics.n, k_hist - idx)), xh_traj[:, 0:idx]))
                    xr_hist = np.hstack((np.zeros((robot.dynamics.n, k_hist - idx)), xr_traj[:, 0:idx]))
                else:
                    xh_hist = xh_traj[:, idx - k_hist:idx]
                    xr_hist = xr_traj[:, idx - k_hist:idx]
                
                # generate safe trajectory for robot given this goal
                safety, ur_traj, distance, new_belief, obs_loc = generate_trajectories(robot, human, r_belief_nominal.belief, traj_horizon, goals, goal, xh_hist, xr_hist, model, stats_file=None, stats=stats, verbose=False, plot=False)

                posts.append(new_belief)
                ur_trajs.append(ur_traj)

        # choose whether to influence or be curteous
        influence_human = False
        if human_waiting:
            influence_human = True
        if influence_human:
            obj = influence_obj
        else:
            obj = courtesy_obj

        if robot_type == "cbp_nn":
            goal_idx = obj(human, robot, goals, r_belief_prior, posts)
            if np.linalg.norm(robot.x[[0,2]] - robot.goal[[0,2]]) >= 1:
                ur = ur_trajs[goal_idx][:,[0]]
            else:
                ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
        elif robot_type == "cbp":
            goal_idx = obj(human, robot, goals, r_belief_prior, posts)
        elif robot_type == "baseline":
            goal_idx = np.linalg.norm(robot.x - goals, axis=0).argmin()
        elif robot_type == "baseline_belief":
            # pick the closest goal that the human is not moving towards
            dists = np.linalg.norm(robot.x - goals, axis=0)
            if r_belief_nominal.belief.max() > 0.8:
                dists[r_belief_nominal.belief.argmax()] = np.inf
            else:
                ur = np.zeros((2,1))
            # dists[r_belief_nominal.belief.argmax()] = np.inf
            goal_idx = dists.argmin()

        # update human's belief
        if np.linalg.norm(ur) > 1e-3:
            human.update_belief(robot.x, ur)
            # set min belief to 0.01
            human.belief.belief = np.maximum(human.belief.belief, 0.01)

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
        r_beliefs_beta = np.dstack((r_beliefs_beta, r_belief_beta.belief))
        r_belief_likelihoods.append(likelihoods)
        # save human's actual intended goal
        h_goal_idxs.append(np.argmin(np.linalg.norm(human.goal - goals, axis=0)))
        # save robot's actual intended goal
        r_goal_idxs.append(np.argmin(np.linalg.norm(robot.goal - goals, axis=0)))
        all_goals = np.vstack((all_goals, goals))

        h_goal_reached = False
        if np.linalg.norm(human.x[[0,2]] - human.goal[[0,2]]) < 0.5:
            h_goal_reached = True
        r_goal_reached = False
        if np.linalg.norm(robot.x[[0,2]] - robot.goal[[0,2]]) < 0.5:
            r_goal_reached = True
        both_at_goal = (h_goal_reached and r_goal_reached) and (np.linalg.norm(human.goal - robot.goal) > 1e-3)

        if plot:
            # plot
            ax.clear()
            overlay_timesteps(ax, xh_traj, xr_traj, n_steps=idx)
            ax.scatter(xh0[0], xh0[2], c="blue")
            ax.scatter(xr0[0], xr0[2], c="red")
            if obs_loc is not None:
                ax.scatter(obs_loc[0], obs_loc[2], c="yellow")
            ax.scatter(goals[0], goals[2], c=goal_colors)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

            h_belief_ax.clear()
            for goal_idx in range(h_beliefs.shape[1]):
                h_belief_ax.plot(h_beliefs[:,goal_idx], label=f"P(g{goal_idx})", c=goal_colors[goal_idx])
            h_belief_ax.set_xlabel("h belief of r")
            h_belief_ax.legend()

            r_belief_ax.clear()
            for goal_idx in range(r_beliefs.shape[1]):
                r_belief_ax.plot(r_beliefs[:,goal_idx], label=f"P(g{goal_idx})", c=goal_colors[goal_idx])
            # plot nomninal belief with dashed lines
            for goal_idx in range(r_beliefs_nominal.shape[1]):
                r_belief_ax.plot(r_beliefs_nominal[:,goal_idx], c=goal_colors[goal_idx], linestyle="--")
            r_belief_ax.set_xlabel("r belief of h")
            r_belief_ax.legend()

            h_goal_ax.clear()
            h_goal_ax.plot(h_goal_idxs, c="blue", label="h goal")
            h_goal_ax.plot(r_goal_idxs, c="red", label="r goal")
            h_goal_ax.legend()

            r_beta_ax.clear()
            # for now, plotting marginalized belief
            for beta_idx in range(r_belief_beta.betas.shape[0]):
                r_beta_ax.plot(r_beliefs_beta[:,beta_idx,:].sum(axis=0), label=f"b={r_belief_beta.betas[beta_idx]}")
            r_beta_ax.legend()

            r_likelihoods_ax.clear()
            l = np.array(r_belief_likelihoods)
            for theta_idx in range(r_belief_nominal.thetas.shape[1]):
                r_likelihoods_ax.plot(l[:,theta_idx], label=f"theta={theta_idx}", c=goal_colors[theta_idx])
            # for theta_idx in range(r_belief_beta.thetas.shape[1]):
            #     r_likelihoods_ax.plot(r_beliefs_beta[theta_idx,:,:].sum(axis=0), label=f"theta={theta_idx}", c=goal_colors[theta_idx])
            r_likelihoods_ax.legend()

            plt.pause(0.01)
    if plot:
        plt.show()

    data = {"xh_traj": xh_traj, "xr_traj": xr_traj, "h_beliefs": h_beliefs, "r_beliefs": r_beliefs, "r_beliefs_nominal": r_beliefs_nominal, "r_beliefs_beta": r_beliefs_beta, "r_belief_likelihoods": r_belief_likelihoods, "h_goal_idxs": h_goal_idxs, "r_goal_idxs": r_goal_idxs, "goals": all_goals, "robot_type": robot_type, "human_type": human_type, "is_robot_waiting": is_robot_waiting, "is_human_waiting": is_human_waiting, "team_score": team_score}

    return data

def run_full_games(filepath="./data/sim_stats.pkl", n_traj=10):
    robot_types = ["cbp_nn", "cbp", "baseline", "baseline_belief"]
    # robot_types = ["cbp_nn"]
    all_stats = {robot_type: [] for robot_type in robot_types}
    for robot_type in robot_types:
        # TODO: standardize initial conditions better (since # goal resamples may change, this way doesn't work)
        np.random.seed(0)
        for i in tqdm(range(n_traj)):
            if i % 2 == 0:
                human_type = "stubborn"
            else:
                human_type = "waiting"
            data = run_full_game(robot_type=robot_type, human_type=human_type, plot=False)
            all_stats[robot_type].append(data)
    # save all_stats
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(all_stats, f)

if __name__ == "__main__":
    # np.random.seed(0)
    # run_simulation(robot_type="baseline_belief", human_type="moving", plot=True)
    # run_simulation(robot_type="cbp", human_type="moving", plot=True)
    # run_simulation_nn(robot_type="cbp_nn", human_type="moving", plot=True)

    # filepath = f"./data/cbp_sim/cbp_compare_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    # n_traj = 10
    # run_simulations(filepath, n_traj=n_traj)

    # np.random.seed(0)
    # run_full_game(robot_type="cbp", human_type="stubborn", plot=True)

    filepath = f"./data/cbp_sim/cbp_full_game_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    n_traj = 100
    run_full_games(filepath, n_traj=n_traj)
    