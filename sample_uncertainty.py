# randomly sampling inputs that result in uncertain predictions from intention prediction model

import numpy as np
import torch
import torch.nn as nn
softmax = torch.nn.Softmax(dim=1)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_predictor import create_model
from intention_utils import initialize_problem, overlay_timesteps, get_robot_plan

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def max_entropy(probs):
    logits = torch.log2(probs)
    entropy = -torch.sum(probs * logits, dim=1).detach().numpy()
    entropy = np.nan_to_num(entropy, nan=-np.inf)

    max_ent_idx = np.argmax(entropy)
    return max_ent_idx

def min_entropy(probs):
    logits = torch.log2(probs)
    entropy = -torch.sum(probs * logits, dim=1).detach().numpy()
    entropy = np.nan_to_num(entropy, nan=np.inf)

    min_ent_idx = np.argmin(entropy)
    return min_ent_idx

def zero_prob(probs):
    # find the index that gives lowest probability on any one goal
    pp = probs.detach().numpy()
    return np.argmin(np.min(pp, axis=1))

def max_goal(probs, idx=0):
    # find the index that gives the highest probability on goal 0
    pp = probs.detach().numpy()
    return np.argmax(pp[:,idx])

def min_goal(probs, idx=0):
    pp = probs.detach().numpy()
    return np.argmin(pp[:,idx])

def random_samples(robot, u_max, k_plan, n_sample):
    # returns randomly sampled plans for the robot
    sampled_u = np.random.uniform(low=-u_max, high=u_max, size=(robot.dynamics.m, k_plan, n_sample))
    x0 = robot.x

    x_samples = np.zeros((robot.dynamics.n, k_plan, n_sample))
    # create state trajectories from random controls
    for j in range(k_plan):
        x0 = (robot.dynamics.A @ x0) + (robot.dynamics.B @ sampled_u[:,j,:])
        x_samples[:,j,:] = x0

    xr_plan = torch.tensor(x_samples).float().transpose(0, 2)

    return sampled_u, xr_plan

def goal_plans(robot, u_max, k_plan):
    # return one plan for reaching towards each goal
    xr_plans = []
    ur_plans = []
    for i in range(robot.goals.shape[1]):
        goal = robot.goals[:,[i]]
        x0 = robot.x.copy()
        xr_plan, ur_plan = get_robot_plan(robot=robot, horizon=k_plan, xr0=x0, goal=goal, return_controls=True)
        xr_plans.append(xr_plan)
        ur_plans.append(ur_plan)

    return np.dstack(ur_plans), torch.tensor(np.dstack(xr_plans)).float().transpose(0, 2)

def sample_xr_plan(execute_sample=True, objective_fn=max_entropy, xr_sampler=random_samples, random_goals=False):
    ts = 0.05
    horizon = 100
    k_hist = 5
    k_plan = 20
    u_max = 10

    model = create_model(horizon_len=k_plan)
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor_bayes.pt", map_location=device))

    np.random.seed(1)
    torch.manual_seed(1)
     # creating human and robot
    xh0 = np.array([[0, 0.0, -5, 0.0]]).T
    xr0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T

    if random_goals:
        goals = np.random.uniform(low=-9.0, high=9.0, size=(4,3))
        goals[[1,3],:] = np.zeros((2, 3))
    else:
        goals = np.array([
            [5.0, 0.0, 0.0, 0.0],
            [-5.0, 0.0, 5.0, 0.0],
            [5.0, 0.0, 5.0, 0.0],
        ]).T
    
    r_goal = goals[:,[0]]

    h_dynamics = DIDynamics(ts=ts)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=1)

    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    robot.set_goals(goals)

    # forward simulate 5 timesteps to pass in data to this problem
    xh_traj = np.zeros((4, horizon))
    xr_traj = np.zeros((4, horizon))
    h_goals = np.zeros((4, horizon))
    h_goal_reached = np.zeros((1, horizon))
    xr_plans = np.zeros((4, k_plan, horizon))

    fig, ax = plt.subplots()

    for i in range(horizon):
        # save data
        xh_traj[:,[i]] = human.x
        xr_traj[:,[i]] = robot.x
        h_goals[:,[i]] = human.get_goal()
        # check if human reached its goal
        if np.linalg.norm(human.x - human.get_goal()) < 0.1:
            h_goal_reached[:,i] = 1

        if i > k_hist:
            xh_hist = xh_traj[:,i-k_hist+1:i+1]
            xr_hist = xr_traj[:,i-k_hist+1:i+1]
            
            sampled_u, xr_plan = xr_sampler(robot, u_max, k_plan)
            n_sample = sampled_u.shape[2]
            traj_hist = torch.tensor(np.hstack((xh_hist.T, xr_hist.T))).float().unsqueeze(0).repeat(n_sample, 1, 1)
            goals_in = torch.tensor(goals).float().unsqueeze(0).repeat(n_sample, 1, 1)

            model_outs = model(traj_hist, xr_plan, goals_in)
            probs = softmax(model_outs)

            optimal_plan_idx = objective_fn(probs)
            print(probs[optimal_plan_idx].detach().numpy())
            xr_plans[:,:,i] = xr_plan[optimal_plan_idx,:,:].T
            ur_plan = sampled_u[:,:,optimal_plan_idx]

            # plot optimized plan (if execute_sample is False)
            if not execute_sample:
                overlay_timesteps(ax, xh_hist, xr_hist, goals)
                overlay_timesteps(ax, [], xr_plans[:,:,i], goals)

        # take step
        uh = human.get_u(robot.x)
        if execute_sample:
            if i <= k_hist:
                ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
            else:
                ur = ur_plan[:,[0]]
        else:
            # set ur to be goal-directed action
            ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # update human's belief (if applicable)
        if type(human) == BayesHuman:
            human.update_belief(robot.x, ur)

        xh = human.step(uh)
        xr = robot.step(ur)
    
     # plot trajectory
    overlay_timesteps(ax, xh_traj, xr_traj, goals)

    # for i in range(k_hist, horizon):
    #     # robot planned trajectory
    #     xr_plan = xr_plans[:,:,i]
    #     points = xr_plan[[0,2],:].T.reshape(-1, 1, 2)
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)

    #     norm = plt.Normalize(0, k_plan)
    #     lc = LineCollection(segments, cmap='Purples', norm=norm)
    #     # Set the values used for colormapping
    #     lc.set_array(np.arange(k_plan+1))
    #     lc.set_linewidth(2)
    #     line = ax.add_collection(lc)
    plt.show()
    
if __name__ == "__main__":
    n_sample = 1000
    # xr_sampler = lambda robot, u_max, k_plan: random_samples(robot, u_max, k_plan, n_sample=n_sample)
    xr_sampler = goal_plans
    # objective_fn = zero_prob
    # objective_fn = lambda x: min_goal(x, idx=0)
    objective_fn = max_entropy
    random_goals = True

    sample_xr_plan(execute_sample=True, objective_fn=objective_fn, xr_sampler=xr_sampler, random_goals=random_goals)
