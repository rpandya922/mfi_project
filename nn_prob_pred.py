import numpy as np
import matplotlib.pyplot as plt
import torch
softmax = torch.nn.Softmax(dim=1)
from tqdm import tqdm
import pickle
import os
import h5py

from intention_predictor import create_model
from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from human import Human
from robot import Robot
from intention_utils import overlay_timesteps, get_robot_plan, process_model_input

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_trajectory(human, robot, goals, horizon=None, model=None, plot=True, robot_obs=None, k_hist=5, k_plan=20):
    """
    inputs:
        human: Human object
        robot: Robot object
        goals: 3xN array of goals
        horizon: number of timesteps to run for (if None, run until human reaches goal)
        model: model to use for predicting human intention
        plot: whether to plot the trajectory
        robot_obs: optional synthetic obstacle for the robot to avoid
    outputs:
        xh_traj: 3xT array of human states
        xr_traj: 3xT array of robot states
        all_r_beliefs: 3x3xT array of robot beliefs about human intention
        h_goal_reached: index of goal reached by human
    """
    # show both agent trajectories when the robot reasons about its effect on the human
    if plot:
        # create figure with equal aspect ratio
        # fig, ax = plt.subplots()
        # ax.set_aspect('equal')
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        axes = np.array(axes).flatten()
        ax = axes[0]
        # make ax equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        h_belief_ax = axes[1]
        r_belief_ax = axes[2]
        goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
        h_goal_ax = axes[3]
    xh = human.x
    xr = robot.x

    # if no horizon is given, run until the human reaches the goal (but set arrays to 100 and expand arrays if necessary)
    if horizon is None:
        fixed_horizon = False
        horizon = np.inf
        arr_size = 100
    else:
        fixed_horizon = True
        arr_size = horizon

    # save robot predictions if model is passed in
    if model is not None:
        all_r_beliefs = np.zeros((goals.shape[1], goals.shape[1], arr_size))
    else:
        all_r_beliefs = None
    if type(human) == BayesHuman:
        all_h_beliefs = np.zeros((goals.shape[1], arr_size))
    else:
        all_h_beliefs = None
    xh_traj = np.zeros((human.dynamics.n, arr_size+1))
    xr_traj = np.zeros((robot.dynamics.n, arr_size+1))
    xh_traj[:,[0]] = xh
    xr_traj[:,[0]] = xr
    h_goal_reached = -1

    r_goal_idx = np.argmin(np.linalg.norm(goals[[0,2]] - robot.goal[[0,2]], axis=0))

    i = 0
    while i < horizon:
        if plot:
            # plotting
            ax.cla()
            # plot trajectory trail so far
            overlay_timesteps(ax, xh_traj[:,0:i], xr_traj[:,0:i], n_steps=i)

            # TODO: highlight robot's predicted goal of the human
            if i > 0:
                # get the previous robot belief
                r_belief = all_r_beliefs[:,:,i-1][r_goal_idx]
                # plot a transparent circle on each of the goals with radius proportional to the robot's belief
                for goal_idx in range(goals.shape[1]):
                    goal = goals[:,[goal_idx]]
                    # plot a circle with radius proportional to the robot's belief
                    ax.add_artist(plt.Circle((goal[0], goal[2]), r_belief[goal_idx]*2, color=goal_colors[goal_idx], alpha=0.3))

            ax.scatter(goals[0], goals[2], c=goal_colors, s=100)
            ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
            ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            # img_path = f"./data/videos/mfi_demo/frames3"
            # img_path += f"/{i:03d}.png"
            # plt.savefig(img_path, dpi=300)

            h_belief_ax.clear()
            h_belief_ax.plot(all_h_beliefs[0,:i], label="P(g0)", c=goal_colors[0])
            h_belief_ax.plot(all_h_beliefs[1,:i], label="P(g1)", c=goal_colors[1])
            h_belief_ax.plot(all_h_beliefs[2,:i], label="P(g2)", c=goal_colors[2])
            h_belief_ax.set_xlabel("h belief of r")
            h_belief_ax.legend()

            r_belief = all_r_beliefs[r_goal_idx,:,:i]
            r_belief_ax.clear()
            r_belief_ax.plot(r_belief[0], label="P(g0)", c=goal_colors[0])
            r_belief_ax.plot(r_belief[1], label="P(g1)", c=goal_colors[1])
            r_belief_ax.plot(r_belief[2], label="P(g2)", c=goal_colors[2])

            # plot other goal idx beliefs in dotted lines
            r_belief0 = all_r_beliefs[2,:,:i]
            r_belief_ax.plot(r_belief0[0], label="P(g0)", c=goal_colors[0], linestyle="--")
            r_belief_ax.plot(r_belief0[1], label="P(g1)", c=goal_colors[1], linestyle="--")
            r_belief_ax.plot(r_belief0[2], label="P(g2)", c=goal_colors[2], linestyle="--")
            r_belief_ax.set_xlabel("r belief of h")
            r_belief_ax.legend()

            plt.pause(0.01)

        # compute the robot's prediction of the human's intention
        # but only if model is passed in
        if model is not None:
            if i < k_hist:
                # get zero-padded history of both agents
                xh_hist = np.hstack((np.zeros((human.dynamics.n, k_hist-i)), xh_traj[:,0:i]))
                xr_hist = np.hstack((np.zeros((robot.dynamics.n, k_hist-i)), xr_traj[:,0:i]))
            else:
                xh_hist = xh_traj[:,i-k_hist:i]
                xr_hist = xr_traj[:,i-k_hist:i]
            r_beliefs = []
            for goal_idx in range(robot.goals.shape[1]):
                goal = robot.goals[:,[goal_idx]]
                # compute robot plan given this goal
                xr_plan = get_robot_plan(robot, horizon=k_plan, goal=goal)
                r_beliefs.append(softmax(model(*process_model_input(xh_hist, xr_hist, xr_plan.T, goals))).detach().numpy()[0])
            # import ipdb; ipdb.set_trace()
            # use robot's belief to pick which goal to move toward by picking the one that puts the highest probability on the human's nominal goal (what we observe in the first 5 timesteps)
            r_beliefs = np.array(r_beliefs)
            # TODO: set goal only if specified
            # r_goal_idx = np.argmax(r_beliefs[:,nominal_goal_idx])
            # robot.set_goal(robot.goals[:,[r_goal_idx]])
            all_r_beliefs[:,:,i] = r_beliefs

        # compute agent controls
        uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
        if robot_obs is not None:
            ur += robot.dynamics.get_robot_control(robot.x, robot_obs)
        # robot should stay still for 5 timesteps
        if i < 5:
            ur = np.zeros(ur.shape)

        # update human belief (if applicable)
        if type(human) == BayesHuman and i >= 5:
            human.update_belief(robot.x, ur)

        # compute new states
        xh = human.step(uh)
        xr = robot.step(ur)

        # save data
        if all_h_beliefs is not None:
            all_h_beliefs[:,i] = human.belief.belief
        xh_traj[:,[i+1]] = xh
        xr_traj[:,[i+1]] = xr

        # check if human has reached a goal
        if not fixed_horizon:
            # check if we need to expand arrays
            if i >= arr_size-1:
                arr_size *= 2
                xh_traj = np.hstack((xh_traj, np.zeros((human.dynamics.n, arr_size))))
                xr_traj = np.hstack((xr_traj, np.zeros((robot.dynamics.n, arr_size))))
                if all_h_beliefs is not None:
                    all_h_beliefs = np.hstack((all_h_beliefs, np.zeros((goals.shape[1], arr_size))))
                if all_r_beliefs is not None:
                    all_r_beliefs = np.dstack((all_r_beliefs, np.zeros((goals.shape[1], goals.shape[1], arr_size))))
                # print("expanding arrays")
            goal_dist = np.linalg.norm(xh[[0,2]] - human.goal[[0,2]])
            dists = np.linalg.norm(goals[[0,2]] - xh[[0,2]], axis=0)
            # goal_dist = np.min(dists)
            if goal_dist < 0.5:
                h_goal_reached = np.argmin(dists)
                break
        
        # NOTE: don't put any code in the loop after this
        i += 1
    
    # return only non-zero elements of arrays
    xh_traj = xh_traj[:,0:i+1]
    xr_traj = xr_traj[:,0:i+1]
    if all_r_beliefs is not None:
        all_r_beliefs = all_r_beliefs[:,:,0:i+1]
    return xh_traj, xr_traj, all_r_beliefs, h_goal_reached

def simulate_init_cond(xr0, xh0, human, robot, goals, n_traj=10):
    """
    Runs a number of simulations from the same initial conditions and returns the probability of each goal being the human's intention
    """
    # save data
    all_xh_traj = []
    all_xr_traj = []
    all_goals_reached = []
    goals_reached = np.zeros(goals.shape[1])
    for i in range(n_traj):
        # set the human and robot's initial states
        # TODO: deal with human's belief
        human.x = xh0
        robot.x = xr0
        # randomly set the robot's goal to one of the options
        robot.set_goal(goals[:,[np.random.randint(0, goals.shape[1])]])
        # generate a synthetic obstacle for the robot that lies between the robotand its goal
        # get bounds of the rectangle that contains the robot and its goal
        x_min = min(robot.x[0], robot.goal[0])
        x_max = max(robot.x[0], robot.goal[0])
        y_min = min(robot.x[2], robot.goal[2])
        y_max = max(robot.x[2], robot.goal[2])
        # randomly sample a point in this rectangle
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        robot_obs = np.array([[x[0], 0, y[0], 0]]).T
        # run the simulation
        xh_traj, xr_traj, _, h_goal_reached = run_trajectory(human, robot, goals, plot=False, robot_obs=robot_obs)
        # increment the count for the goal that was reached
        goals_reached[h_goal_reached] += 1
        # save data
        all_xh_traj.append(xh_traj)
        all_xr_traj.append(xr_traj)
        all_goals_reached.append(h_goal_reached)
    # return the probability of each goal being the human's intention
    return all_xh_traj, all_xr_traj, all_goals_reached, goals_reached / n_traj, goals

def simulate_init_cond_branching(xr0, xh0, human, robot, goals, n_traj=10, branching_times=[10, 50, 100]):
    """
    Runs a number of simulations from the same initial conditions and returns the probability of each goal being the human's intention
    """
    # save data
    all_xh_traj = []
    all_xr_traj = []
    all_goals_reached = []
    branching_nums = []
    xh_hists = []
    xr_hists = []
    goals_reached = np.zeros(goals.shape[1])
    # keep track of initial conditions for branching points
    branching_num = 0
    xh_init_hist = np.zeros((4, 5))
    xr_init_hist = np.zeros((4, 5)) # necessary for adding shared history to branching trajectories
    init_conds = [(xh0, xr0, False, np.ones(goals.shape[1]) / goals.shape[1], branching_num, xh_init_hist, xr_init_hist) for _ in range(n_traj)] # start from the first initial contition n_traj times
    branching_num += 1
    # fig, ax = plt.subplots()
    while len(init_conds) > 0:
        # set the human and robot's initial states
        xh, xr, is_branch, h_belief, b_num_curr, xh_hist, xr_hist = init_conds.pop(0)
        human.x = xh
        human.belief.belief = h_belief
        robot.x = xr
        # randomly set the robot's goal to one of the options
        robot.set_goal(goals[:,[np.random.randint(0, goals.shape[1])]])
        # generate a synthetic obstacle for the robot that lies between the robot and its goal
        # get bounds of the rectangle that contains the robot and its goal
        x_min = min(robot.x[0], robot.goal[0])
        x_max = max(robot.x[0], robot.goal[0])
        y_min = min(robot.x[2], robot.goal[2])
        y_max = max(robot.x[2], robot.goal[2])
        # randomly sample a point in this rectangle
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        robot_obs = np.array([[x[0], 0, y[0], 0]]).T
        # run the simulation
        # xh_traj, xr_traj, _, h_goal_reached = run_trajectory(human, robot, goals, plot=False, robot_obs=robot_obs)

        # if no horizon is given, run until the human reaches the goal (but set arrays to 100 and expand arrays if necessary)
        fixed_horizon = False
        horizon = np.inf
        arr_size = 100

        xh_traj = np.zeros((human.dynamics.n, arr_size+1))
        xr_traj = np.zeros((robot.dynamics.n, arr_size+1))
        xh_traj[:,[0]] = human.x
        xr_traj[:,[0]] = robot.x
        h_goal_reached = -1

        r_goal_idx = np.argmin(np.linalg.norm(goals[[0,2]] - robot.goal[[0,2]], axis=0))

        i = 0
        while i < horizon:
             # compute agent controls
            uh = human.get_u(robot.x)
            ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
            if robot_obs is not None:
                ur += robot.dynamics.get_robot_control(robot.x, robot_obs)
            # robot should stay still for 5 timesteps
            if i < 5:
                ur = np.zeros(ur.shape)

            # update human belief (if applicable)
            if type(human) == BayesHuman and i >= 5:
                human.update_belief(robot.x, ur)

            # compute new states
            xh = human.step(uh)
            xr = robot.step(ur)

            # save new initial conditions if we're at a branching point
            if i in branching_times and not is_branch:
                h_belief = human.belief.belief.copy()
                xh_init_hist = xh_traj[:,i-4:i+1]
                xr_init_hist = xr_traj[:,i-4:i+1]
                init_conds = init_conds + [((xh, xr, True, h_belief, branching_num, xh_init_hist, xr_init_hist)) for _ in range(n_traj)]
                branching_num += 1

            # save data
            xh_traj[:,[i+1]] = xh
            xr_traj[:,[i+1]] = xr

            # check if human has reached a goal
            if not fixed_horizon:
                # check if we need to expand arrays
                if i >= arr_size-1:
                    arr_size *= 2
                    xh_traj = np.hstack((xh_traj, np.zeros((human.dynamics.n, arr_size))))
                    xr_traj = np.hstack((xr_traj, np.zeros((robot.dynamics.n, arr_size))))
                    # print("expanding arrays")
                dists = np.linalg.norm(goals[[0,2]] - xh[[0,2]], axis=0)
                if np.min(dists) < 0.5:
                    h_goal_reached = np.argmin(dists)
                    break
            
            # NOTE: don't put any code in the loop after this
            i += 1
        # TODO: fix this, since I think it messes up data labeling (some trajectories start with empirical labels in the middle because of branching)
        # the trajectory may start from branching point, I have decided not to prepend the "parent trajectory" because that just duplicates some data
        xh_traj = xh_traj[:,0:i+1]
        xr_traj = xr_traj[:,0:i+1]

        # increment the count for the goal that was reached
        goals_reached[h_goal_reached] += 1
        # save data
        all_xh_traj.append(xh_traj)
        all_xr_traj.append(xr_traj)
        all_goals_reached.append(h_goal_reached)
        branching_nums.append(b_num_curr)
        xh_hists.append(xh_hist)
        xr_hists.append(xr_hist)

        # (temporary) plotting
        # ax.cla()
        # ax.set_aspect('equal', adjustable='box')
        # # plot trajectory trail so far
        # overlay_timesteps(ax, xh_traj, xr_traj, n_steps=i)
        # ax.scatter(goals[0], goals[2], c="green", s=100)
        # ax.set_xlim(-10, 10)
        # ax.set_ylim(-10, 10)
        # plt.pause(0.01)
    # return the probability of each goal being the human's intention
    # print(np.sum(goals_reached))
    # plt.show()
    # import ipdb; ipdb.set_trace()
    # 1/0
    return all_xh_traj, all_xr_traj, all_goals_reached, goals_reached / np.sum(goals_reached), goals, xh_hists, xr_hists

def compute_features(xh_hist, xr_hist, xr_future, goals):
    """
    features list (for history traj):
        - distance between human and robot
        - distance of human to each goal
        - distance of robot to each goal
        - angle from human's velocity towards each goal
        - angle from robot's velocity towards each goal
    features list (for robot future traj):
        - distance from robot to each goal
        - angle from robot's velocity towards each goal
    """
    hr_dists = np.linalg.norm(xh_hist[[0,2]] - xr_hist[[0,2]], axis=0, keepdims=True).T
    h_goal_dists = np.linalg.norm(xh_hist.T[:,:,None] - goals, axis=1)
    r_goal_dists = np.linalg.norm(xr_hist.T[:,:,None] - goals, axis=1)
    h_rel = (goals - xh_hist.T[:,:,None])[:,[0,2],:].swapaxes(0,1)
    h_vel = xh_hist[[1,3]]
    r_rel = (goals - xr_hist.T[:,:,None])[:,[0,2],:].swapaxes(0,1)
    r_vel = xr_hist[[1,3]]

    # compute angle between human's velocity and vector to each goal
    h_rel_unit = h_rel / np.linalg.norm(h_rel, axis=0)
    # need to handle zero vectors
    h_vel_norm = np.linalg.norm(h_vel, axis=0)
    h_vel_norm[h_vel_norm == 0] = 1
    h_vel_unit = h_vel / h_vel_norm
    h_angles = np.vstack([np.arccos(np.clip(np.dot(h_vel_unit.T, h_rel_unit[:,:,i]).diagonal(), -1.0, 1.0)) for i in range(goals.shape[1])]).T

    # compute angle between robot's velocity and vector to each goal
    r_rel_unit = r_rel / np.linalg.norm(r_rel, axis=0)
    # need to handle zero vectors
    r_vel_norm = np.linalg.norm(r_vel, axis=0)
    r_vel_norm[r_vel_norm == 0] = 1
    r_vel_unit = r_vel / r_vel_norm
    r_angles = np.vstack([np.arccos(np.clip(np.dot(r_vel_unit.T, r_rel_unit[:,:,i]).diagonal(), -1.0, 1.0)) for i in range(goals.shape[1])]).T

    r_future_dists = np.linalg.norm(xr_future.T[:,:,None] - goals, axis=1)
    r_future_rel = (goals - xr_future.T[:,:,None])[:,[0,2],:].swapaxes(0,1)
    r_future_vel = xr_future[[1,3]]
    r_future_rel_unit = r_future_rel / np.linalg.norm(r_future_rel, axis=0)
    # need to handle zero vectors
    r_future_vel_norm = np.linalg.norm(r_future_vel, axis=0)
    r_future_vel_norm[r_future_vel_norm == 0] = 1
    r_future_vel_unit = r_future_vel / r_future_vel_norm
    r_future_angles = np.vstack([np.arccos(np.clip(np.dot(r_future_vel_unit.T, r_future_rel_unit[:,:,i]).diagonal(), -1.0, 1.0)) for i in range(goals.shape[1])]).T

    # concatenate features
    input_feats = np.hstack((hr_dists, h_goal_dists, r_goal_dists, h_angles, r_angles))
    future_feats = np.hstack((r_future_dists, r_future_angles))

    return input_feats.T, future_feats.T

def create_labels(all_xh_traj, all_xr_traj, all_goals_reached, goal_probs, goals, mode="interpolate", history=5, horizon=5, branching=True, n_traj=10, all_xh_hist=None, all_xr_hist=None):

    input_traj = []
    robot_future = []
    input_goals = []
    labels = []

    if branching:
        n_traj_groups = int(len(all_xh_traj) / n_traj)
        for i in range(n_traj_groups):
            traj_group_idxs = range(i*n_traj, (i+1)*n_traj)
            xh_traj_group = [all_xh_traj[j] for j in traj_group_idxs]
            xr_traj_group = [all_xr_traj[j] for j in traj_group_idxs]
            goals_reached_group = [all_goals_reached[j] for j in traj_group_idxs]
            if all_xh_hist is not None:
                xh_hist_group = [all_xh_hist[j] for j in traj_group_idxs]
                xr_hist_group = [all_xr_hist[j] for j in traj_group_idxs]
            else:
                xh_hist_group = None
                xr_hist_group = None
            goal_probs_group = np.zeros(goal_probs.shape)
            for goal_reached in goals_reached_group:
                goal_probs_group[goal_reached] += 1
            goal_probs_group /= n_traj

            it, rf, ig, l = create_labels(xh_traj_group, xr_traj_group, goals_reached_group, goal_probs_group, goals, mode=mode, history=history, horizon=horizon, branching=False, n_traj=n_traj, all_xh_hist=xh_hist_group, all_xr_hist=xr_hist_group)
            input_traj += it
            robot_future += rf
            input_goals += ig
            labels += l

    for i in range(len(all_xh_traj)):
        goal_prob_label = goal_probs.copy()
        xh_traj = all_xh_traj[i]
        xr_traj = all_xr_traj[i]
        goal_reached = all_goals_reached[i]
        
        prob_step_sizes = -goal_prob_label / xh_traj.shape[1]
        prob_step_sizes[goal_reached] = (1 - goal_prob_label[goal_reached]) / xh_traj.shape[1]
        for j in range(xh_traj.shape[1]):
            # set the label for the current timestep
            if mode == "interpolate":
                goal_prob_label += prob_step_sizes

            # zero-pad inputs if necessary
            xh_hist = np.zeros((xh_traj.shape[0], history))
            xr_hist = np.zeros((xr_traj.shape[0], history))
            xr_future = np.zeros((xr_traj.shape[0], horizon))
            
            if j < history:
                xh_hist[:,history-(j+1):] = xh_traj[:,0:j+1] # from run trajectory
                if (all_xh_hist is not None) and (j+1 < history):
                        xh_hist[:,0:history-(j+1)] = all_xh_hist[i][:,-history+(j+1):] # from branching
                xr_hist[:,history-(j+1):] = xr_traj[:,0:j+1] # from run trajectory
                if (all_xr_hist is not None) and (j+1 < history):
                    xr_hist[:,0:history-(j+1)] = all_xr_hist[i][:,-history+(j+1):] # from branching
            else:
                xh_hist = xh_traj[:,(j+1)-history:j+1]
                xr_hist = xr_traj[:,(j+1)-history:j+1]

            if j + horizon >= xh_traj.shape[1]:
                xr_future[:,0:xh_traj.shape[1]-(j+1)] = xr_traj[:,j+1:]
            else:
                xr_future = xr_traj[:,j+1:(j+1)+horizon]

            # add data point to dataset
            input_feats, future_feats = compute_features(xh_hist, xr_hist, xr_future, goals)
            # LSTM expects input of size (sequence length, # features) [batch size dealt with separately]
            input_traj.append(torch.tensor(np.vstack((xh_hist, xr_hist, input_feats)).T).float()) # shape (history,n_features)
            robot_future.append(torch.tensor(np.vstack((xr_future, future_feats)).T).float()) # shape (horizon,n_future_features)
            input_goals.append(torch.tensor(goals).float()) # shape (n,#goals)
            labels.append(torch.tensor(goal_prob_label).float()) # shape (#goals,)

            # TODO: decide on the proper label for these data points then add them back
            # add a data point that has xr_future zero'd out and uses initial goal probs as labels
            # input_traj.append(torch.tensor(np.vstack((xh_hist, xr_hist)).T).float()) # shape (5,8)
            # robot_future.append(torch.tensor(np.zeros_like(xr_future).T).float()) # shape (5,4)
            # input_goals.append(torch.tensor(goals).float()) # shape (4,3)
            # labels.append(torch.tensor(goal_probs).float()) # shape (3,)

    return input_traj, robot_future, input_goals, labels

def create_dataset(n_init_cond=200, branching=True, n_traj=10):
    # generate n_init_cond initial conditions and find goal_probs for each

    # saving raw data
    xh_traj = []
    xr_traj = []
    goals_reached = []
    goal_probs = []
    all_goals = []
    xh_hist = []
    xr_hist = []
    for i in tqdm(range(n_init_cond)):
        # generate initial conditions
        ts = 0.05
        xh0 = np.random.uniform(-10, 10, (4, 1))
        xh0[[1,3]] = 0
        xr0 = np.random.uniform(-10, 10, (4, 1))
        xr0[[1,3]] = 0
        goals = np.random.uniform(-10, 10, (4, 3))
        goals[[1,3]] = 0
        r_goal = goals[:,[0]] # this is arbitrary since it'll be changed in simulations later anyways

        # create human and robot objects
        W = np.diag([0.0, 0.7, 0.0, 0.7])
        h_dynamics = DIDynamics(ts=ts, W=W)
        r_dynamics = DIDynamics(ts=ts)
        r_dynamics.gamma = 10 # for synthetic obstacles for the robot

        belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
        human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=5)
        # human = Human(xh0, h_dynamics, goals, gamma=5)
        robot = Robot(xr0, r_dynamics, r_goal, dmin=3)

        # simulate trajectories
        if not branching:
            data = simulate_init_cond(xr0, xh0, human, robot, goals, n_traj=n_traj)
        else:
            data = simulate_init_cond_branching(xr0, xh0, human, robot, goals, n_traj=n_traj, branching_times=[20, 40, 60, 80, 100])
        xh_traj.append(data[0])
        xr_traj.append(data[1])
        goals_reached.append(data[2])
        goal_probs.append(data[3])
        all_goals.append(data[4])
        if len(data) > 5:
            xh_hist.append(data[5])
            xr_hist.append(data[6])
    if xh_hist == []:
        xh_hist = None
        xr_hist = None
    return xh_traj, xr_traj, goals_reached, goal_probs, all_goals, xh_hist, xr_hist

def save_data(dataset, path="./data/simulated_interactions_bayes_prob_train.pkl", branching=True, n_traj=10):
    # check if path exists
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if len(dataset) > 5 and dataset[5] is not None:
        with open(path, "wb") as f:
            pickle.dump({"xh_traj": dataset[0], "xr_traj": dataset[1], "goals_reached": dataset[2], "goal_probs": dataset[3], "goals": dataset[4], "branching": branching, "n_traj": n_traj, "xh_hists": dataset[5], "xr_hists": dataset[6]}, f)
    else:
        with open(path, "wb") as f:
            pickle.dump({"xh_traj": dataset[0], "xr_traj": dataset[1], "goals_reached": dataset[2], "goal_probs": dataset[3], "goals": dataset[4], "branching": branching, "n_traj": n_traj}, f)
    print(f"saved raw data to {path}")

def process_and_save_data(raw_data_path, processed_data_path, history=5, horizon=20):
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)
    xh_traj = raw_data["xh_traj"]
    xr_traj = raw_data["xr_traj"]
    goals_reached = raw_data["goals_reached"]
    goal_probs = raw_data["goal_probs"]
    goals = raw_data["goals"]
    branching = raw_data["branching"]
    n_traj = raw_data["n_traj"]
    if "xh_hists" in raw_data:
        xh_hists = raw_data["xh_hists"]
        xr_hists = raw_data["xr_hists"]
    else:
        xh_hists = None
        xr_hists = None

    input_traj = []
    robot_future = []
    input_goals = []
    labels = []

    n_traj_total = len(xh_traj)
    for i in tqdm(range(n_traj_total)):
        if xh_hists is not None:
            xh_hist_i = xh_hists[i]
            xr_hist_i = xr_hists[i]
        else:
            xh_hist_i = None
            xr_hist_i = None
        it, rf, ig, l = create_labels(xh_traj[i], xr_traj[i], goals_reached[i], goal_probs[i], goals[i], history=history, horizon=horizon, branching=branching, n_traj=n_traj, all_xh_hist=xh_hist_i, all_xr_hist=xr_hist_i)
        input_traj += it
        robot_future += rf
        input_goals += ig
        labels += l

    # save in pkl file
    if not os.path.exists(os.path.dirname(processed_data_path)):
        os.makedirs(os.path.dirname(processed_data_path))
    with open(processed_data_path, "wb") as f:
        pickle.dump({"input_traj": torch.stack(input_traj), "robot_future": torch.stack(robot_future), "input_goals": torch.stack(input_goals), "labels": torch.stack(labels)}, f)
    print(f"saved processed data to {processed_data_path}")

def plot_model_pred(model_path):
    # load model
    horizon = 20
    hidden_size = 128
    num_layers = 2
    model = create_model(horizon_len=horizon, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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
    W = np.diag([0.0, 0.7, 0.0, 0.7])
    h_dynamics = DIDynamics(ts=ts, W=W)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=5)
    # human = Human(xh0, h_dynamics, goals, gamma=5)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    robot.set_goals(goals)

    run_trajectory(human, robot, goals, model=model, plot=True)

def save_dataset(raw_data_path="./data/prob_pred/bayes_prob.pkl", processed_data_path="./data/prob_pred/bayes_prob_processed.pkl", n_init_cond=800, branching=True, n_traj=10, history=5, horizon=20):
    dataset = create_dataset(n_init_cond=n_init_cond, branching=branching, n_traj=n_traj)
    save_data(dataset, path=raw_data_path, branching=branching, n_traj=n_traj)
    process_and_save_data(raw_data_path, processed_data_path, history=history, horizon=horizon)

def save_dataset_h5(raw_data_path, processed_h5_path, n_init_cond=800, branching=True, n_traj=10, history=5, horizon=20):
    dataset = create_dataset(n_init_cond=n_init_cond, branching=branching, n_traj=n_traj)
    save_data(dataset, path=raw_data_path, branching=branching, n_traj=n_traj)
    convert_raw_to_h5(raw_data_path)
    process_and_save_h5(raw_data_path.replace(".pkl", ".h5"), processed_h5_path, history=history, horizon=horizon)

def convert_raw_to_h5(raw_data_path):
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)
    xh_traj = raw_data["xh_traj"]
    xr_traj = raw_data["xr_traj"]
    goals_reached = raw_data["goals_reached"]
    goal_probs = raw_data["goal_probs"]
    goals = raw_data["goals"]
    branching = raw_data["branching"]
    n_traj = raw_data["n_traj"]
    if "xh_hists" in raw_data:
        xh_hists = raw_data["xh_hists"]
        xr_hists = raw_data["xr_hists"]
    else:
        xh_hists = None
        xr_hists = None

    # save in h5 file
    h5_path = raw_data_path.replace(".pkl", ".h5")
    with h5py.File(h5_path, "w") as f:
        f.attrs["branching"] = branching
        f.attrs["n_traj"] = n_traj
        f.attrs["n_init"] = len(xh_traj)
        for i in tqdm(range(len(xh_traj))):
            init_grp = f.create_group(f"init_{i}")
            init_grp.create_dataset("goals_reached", data=goals_reached[i])
            init_grp.create_dataset("goal_probs", data=goal_probs[i])
            init_grp.create_dataset("goals", data=goals[i])
            init_grp.attrs["n_traj_init"] = len(xh_traj[i])
            for j in range(len(xh_traj[i])):
                init_grp.create_dataset(f"xh_traj_{j}", data=xh_traj[i][j])
                init_grp.create_dataset(f"xr_traj_{j}", data=xr_traj[i][j])
                if xh_hists is not None:
                    init_grp.create_dataset(f"xh_hist_{j}", data=xh_hists[i][j])
                    init_grp.create_dataset(f"xr_hist_{j}", data=xr_hists[i][j])
    print(f"converted to h5 file {h5_path}")

def process_and_save_h5(raw_data_path, processed_data_path, history=5, horizon=20):
    raw_data = h5py.File(raw_data_path, "r")
    branching = raw_data.attrs["branching"]
    n_traj = raw_data.attrs["n_traj"]
    n_init = raw_data.attrs["n_init"]

    with h5py.File(processed_data_path, "w") as f:
        for i_init in tqdm(range(n_init)):
            # load all trajectories for this initial condition
            grp = raw_data[f"init_{i_init}"]
            n_traj_init = grp.attrs["n_traj_init"]
            xh_traj = [grp[f"xh_traj_{j}"] for j in range(n_traj_init)]
            xr_traj = [grp[f"xr_traj_{j}"] for j in range(n_traj_init)]
            if f"xh_hist_{0}" in grp:
                xh_hist = [grp[f"xh_hist_{j}"] for j in range(n_traj_init)]
                xr_hist = [grp[f"xr_hist_{j}"] for j in range(n_traj_init)]
            else:
                xh_hist = None
                xr_hist = None
            goals_reached = grp["goals_reached"][:]
            goal_probs = grp["goal_probs"][:]
            goals = grp["goals"][:]

            it, rf, ig, l = create_labels(xh_traj, xr_traj, goals_reached, goal_probs, goals, history=history, horizon=horizon, branching=branching, n_traj=n_traj, all_xh_hist=xh_hist, all_xr_hist=xr_hist)
            it = torch.stack(it)
            rf = torch.stack(rf)
            ig = torch.stack(ig)
            l = torch.stack(l)

            if i_init == 0:
                # create datasets
                input_traj = f.create_dataset("input_traj", data=it, maxshape=(None, it.shape[1], it.shape[2]))
                robot_future = f.create_dataset("robot_future", data=rf, maxshape=(None, rf.shape[1], rf.shape[2]))
                input_goals = f.create_dataset("input_goals", data=ig, maxshape=(None, ig.shape[1], ig.shape[2]))
                labels = f.create_dataset("labels", data=l, maxshape=(None, l.shape[1]))
            else:
                # append to datasets
                input_traj.resize((input_traj.shape[0] + it.shape[0]), axis=0)
                input_traj[-it.shape[0]:] = it
                robot_future.resize((robot_future.shape[0] + rf.shape[0]), axis=0)
                robot_future[-rf.shape[0]:] = rf
                input_goals.resize((input_goals.shape[0] + ig.shape[0]), axis=0)
                input_goals[-ig.shape[0]:] = ig
                labels.resize((labels.shape[0] + l.shape[0]), axis=0)
                labels[-l.shape[0]:] = l
    print(f"saved processed data to {processed_data_path}")

def visualize_dataset(raw_data_path):
    # open h5 file
    raw_data = h5py.File(raw_data_path, "r")
    # NOTE: only run this locally so we don't try to transfer the whole dataset over ssh
    n_traj = raw_data.attrs["n_traj"]
    n_init = raw_data.attrs["n_init"]

    for i_init in tqdm(range(n_init)):
        # load all trajectories for this initial condition
        grp = raw_data[f"init_{i_init}"]
        n_traj_init = grp.attrs["n_traj_init"]
        xh_traj = [grp[f"xh_traj_{j}"] for j in range(n_traj_init)]
        xh0 = xh_traj[0][:,0]
        xr_traj = [grp[f"xr_traj_{j}"] for j in range(n_traj_init)]
        xr0 = xr_traj[0][:,0]
        goals_reached = grp["goals_reached"][:]
        goal_probs = grp["goal_probs"][:]
        goals = grp["goals"][:]

        for goal_idx in range(3):
            # get all trajectories that reached this goal
            xh_traj_goal = [xh_traj[j] for j in range(n_traj_init) if goals_reached[j] == goal_idx]
            xr_traj_goal = [xr_traj[j] for j in range(n_traj_init) if goals_reached[j] == goal_idx]

            # plot trajectories
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')
            for i in range(len(xh_traj_goal)):
                overlay_timesteps(ax, xh_traj_goal[i], xr_traj_goal[i], alpha=0.7, n_steps=xh_traj_goal[i].shape[1])
            ax.scatter(goals[0], goals[2], c="green", s=100)
            # plot initial human and robot state
            ax.scatter(xh0[0], xh0[2], c="blue", s=100)
            ax.scatter(xr0[0], xh0[2], c="red", s=100)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            plt.pause(0.01)

        import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # # save_dataset()
    # np.random.seed(2)
    # model_path = "./data/models/prob_pred_intention_predictor_bayes_20230620-205847.pt"
    # model_path = "./data/prob_pred/checkpoints/2023-06-15_13-33-40_lr_0.001_bs_256/model_4.pt"
    # # model_path = "./data/models/sim_intention_predictor_bayes_ll.pt"
    # plot_model_pred(model_path)
    # plt.show()

    # save_dataset()

    # raw_data_path = "./data/prob_pred/bayes_prob_branching.pkl"
    # processed_data_path = "./data/prob_pred/bayes_prob_branching_processed_feats.pkl"
    # save_dataset(raw_data_path, processed_data_path, n_init_cond=400, branching=True, n_traj=10, history=5, horizon=20)
    # process_and_save_data(raw_data_path, processed_data_path, history=5, horizon=20)

    # dataset = create_dataset(n_init_cond=10, branching=True, n_traj=10)
    # save_data(dataset, path=raw_data_path, branching=True, n_traj=10)
    # convert_raw_to_h5(raw_data_path)
    # raw_data_path = "./data/prob_pred/bayes_prob_branching.h5"
    # processed_data_path = "./data/prob_pred/bayes_prob_branching_processed_feats.h5"
    # process_and_save_h5(raw_data_path, processed_data_path, history=5, horizon=20)

    # raw_data_path = "./data/prob_pred/bayes_prob_branching.h5"
    # visualize_dataset(raw_data_path)

    # save new data and convert to h5 with featurization
    raw_data_path = "./data/prob_pred/bayes_prob_branching.pkl"
    processed_data_path = "./data/prob_pred/bayes_prob_branching_processed_feats.h5"
    save_dataset_h5(raw_data_path, processed_data_path, n_init_cond=400, branching=True, n_traj=15, history=5, horizon=20)
