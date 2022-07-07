# randomly sampling inputs that result in uncertain predictions from intention prediction model

import numpy as np
import torch
import torch.nn as nn
softmax = torch.nn.Softmax(dim=1)

from dynamics import DIDynamics
from human import Human
from robot import Robot
from intention_predictor import create_model
from test_predictor import get_robot_plan
from optimize_uncertainty import initialize_problem, overlay_timesteps

def sample_xr_plan():
    ts = 0.05
    horizon = 100
    k_hist = 5
    k_plan = 20
    n_sample = 10000
    u_max = 10

    model = create_model(horizon_len=k_plan)
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor_plan20.pt"))

    np.random.seed(0)
    human, robot, goals = initialize_problem()

    # forward simulate 5 timesteps to pass in data to this problem
    xh_traj = np.zeros((4, horizon))
    xr_traj = np.zeros((4, horizon))
    h_goals = np.zeros((4, horizon))
    h_goal_reached = np.zeros((1, horizon))

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

            sampled_u = np.random.uniform(low=-u_max, high=u_max, size=(robot.dynamics.m, k_plan, n_sample))
            x0 = robot.x

            x_samples = np.zeros((robot.dynamics.n, k_plan, n_sample))
            # create state trajectories from random controls
            for j in range(k_plan):
                x0 = (robot.dynamics.A @ x0) + (robot.dynamics.B @ sampled_u[:,j,:])
                x_samples[:,j,:] = x0

            xr_plan = torch.tensor(x_samples).float().transpose(0, 2)
            traj_hist = torch.tensor(np.hstack((xh_hist.T, xr_hist.T))).float().unsqueeze(0).repeat(n_sample, 1, 1)
            goals_in = torch.tensor(goals).float().unsqueeze(0).repeat(n_sample, 1, 1)

            model_outs = model(traj_hist, xr_plan, goals_in)
            probs = softmax(model_outs)
            logits = torch.log2(probs)
            entropy = -torch.sum(probs * logits, dim=1).detach().numpy()
            entropy = np.nan_to_num(entropy, nan=-np.inf)
            
            max_ent_idx = np.argmax(entropy)
            print(probs[max_ent_idx].detach().numpy())

        # take step
        uh = human.get_u(robot.x)
        if i == 0:
            ur = robot.get_u(human.x, robot.x, human.x)
        else:
            ur = robot.get_u(human.x, xr_traj[:,[i-1]], xh_traj[:,[i-1]])

        xh = human.step(uh)
        xr = robot.step(ur)


def sample_full_traj():
    ts = 0.05
    horizon = 100
    k_hist = 5
    k_plan = 20
    n_sample = 10000
    u_max = 10

    model = create_model(horizon_len=k_plan)
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor_plan20.pt"))

    np.random.seed(0)
    human, robot, goals = initialize_problem()

    # forward simulate 5 timesteps to pass in data to this problem
    xh_traj = np.zeros((4, horizon))
    xr_traj = np.zeros((4, horizon))
    h_goals = np.zeros((4, horizon))
    h_goal_reached = np.zeros((1, horizon))
