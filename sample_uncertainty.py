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
from test_predictor import get_robot_plan
from optimize_uncertainty import initialize_problem, overlay_timesteps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_xr_plan():
    ts = 0.05
    horizon = 25
    k_hist = 5
    k_plan = 20
    n_sample = 10000
    u_max = 10

    model = create_model(horizon_len=k_plan)
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor_bayes.pt", map_location=device))

    np.random.seed(0)
    torch.manual_seed(0)
     # creating human and robot
    xh0 = np.array([[0, 0.0, -5, 0.0]]).T
    xr0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T

    goals = np.array([
        [5.0, 0.0, 0.0, 0.0],
        [-5.0, 0.0, 5.0, 0.0],
        [5.0, 0.0, 5.0, 0.0],
    ]).T
    r_goal = goals[:,[0]]

    h_dynamics = DIDynamics(ts=ts)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=20)
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
            
            # TODO: print the range of entropy

            max_ent_idx = np.argmax(entropy)
            print(probs[max_ent_idx].detach().numpy())
            xr_plans[:,:,i] = xr_plan[max_ent_idx,:,:].T
            ur_plan = sampled_u[:,:,max_ent_idx]

        # take step
        uh = human.get_u(robot.x)
        if i == 0:
            ur = robot.get_u(human.x, robot.x, human.x)
        elif i <= k_hist:
            ur = robot.get_u(human.x, xr_traj[:,[i-1]], xh_traj[:,[i-1]])
        else:
            ur = ur_plan[:,[0]]

        # update human's belief (if applicable)
        if type(human) == BayesHuman:
            human.update_belief(robot.x, ur)

        xh = human.step(uh)
        xr = robot.step(ur)
    
     # plot trajectory
    overlay_timesteps(ax, xh_traj, xr_traj, goals)

    for i in range(k_hist, horizon):
        # robot planned trajectory
        xr_plan = xr_plans[:,:,i]
        points = xr_plan[[0,2],:].T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, k_plan)
        lc = LineCollection(segments, cmap='Purples', norm=norm)
        # Set the values used for colormapping
        lc.set_array(np.arange(k_plan+1))
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    plt.show()

def sample_full_traj():
    # NOTE: unfinished
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

if __name__ == "__main__":
    sample_xr_plan()
