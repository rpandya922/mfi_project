# generate path with multiple robot goals and multiple synthetic obstacles
# use bayesian belief for initial belief_h (that doesn't depend on robot's goal selection)
# feed generated trajectories back to NN to verify safety and efficiency + human belief change + robot goal change


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
from tqdm import tqdm
import pickle
import os
from nn_prob_pred import *

import numpy as np
from numpy.linalg import norm
from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_predictor import create_model
from intention_utils import initialize_problem, overlay_timesteps, get_robot_plan
# from examine_safety import check_safety
from cbp_model import *
import math
import copy
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_safety(robot, human, belief_h, horizon, goals, r_goal, ur_traj, plot=False):
    # given belief from NN, check safety of a robot trajectory
    # d: safe distance between robot and human

    d = 1

    safety = True

    x_r = robot.x
    x_h = human.x
    robot1 = copy.deepcopy(robot)
    human1 = copy.deepcopy(human)
    human2 = copy.deepcopy(human)
    human3 = copy.deepcopy(human)

    human1.goal = goals[:, [0]]
    human2.goal = goals[:, [1]]
    human3.goal = goals[:, [2]]
    # robot1.goal = goals[:, [1]] # robot1's goal is robot's goal
    robot1.goal = r_goal

    # print('human2.goal: ')
    # print(human2.goal)

    xr = x_r
    xh1 = x_h
    xh2 = x_h
    xh3 = x_h

    p1 = belief_h[0]
    p2 = belief_h[1]
    p3 = belief_h[2]

    for i in range(horizon):

        sigma0 = 0.1  # initial std for uncertainty
        sigma = np.sqrt(i) * sigma0

        if plot:
            # plotting
            ax.cla()
            # plot trajectory trail so far
            overlay_timesteps(ax, xh1_traj[:, 0:i], xr1_traj[:, 0:i], n_steps=i)

            # TODO: highlight robot's predicted goal of the human
            ax.scatter(obs[0, 0], obs[0, 2], c="gold", s=100)
            ax.scatter(goals[0], goals[2], c="green", s=100)
            ax.scatter(human1.x[0], human1.x[2], c="#034483", s=100)
            ax.scatter(human1.x[0], human1.x[2], c="#034483", s=100 + 50 * i, alpha=p1, edgecolors='none')
            ax.scatter(human2.x[0], human2.x[2], c="#034483", s=100)
            ax.scatter(human2.x[0], human2.x[2], c="#034483", s=100 + 50 * i, alpha=p2, edgecolors='none')
            ax.scatter(human3.x[0], human3.x[2], c="#034483", s=100)
            ax.scatter(human3.x[0], human3.x[2], c="#034483", s=100 + 50 * i, alpha=p3, edgecolors='none')
            ax.scatter(robot1.x[0], robot1.x[2], c="#800E0E", s=100)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 15)
            img_path = f"./data/videos/mfi_demo/generate_traj_loop"
            img_path += f"/{i:03d}.png"
            plt.savefig(img_path, dpi=300)


        uh1 = human1.dynamics.get_goal_control(human1.x, human1.goal)  # without avoid robot control
        # human.dynamics.gamma = 1  # default value
        human1.dynamics.gamma = 10  # default value
        uh1 = uh1 + human1.dynamics.get_robot_control(human1.x,
                                                      robot1.x)  # potential field control to avoid robot, parameter gamma to tune the aggressiveness

        uh2 = human2.dynamics.get_goal_control(human2.x, human2.goal)  # without avoid robot control
        # human.dynamics.gamma = 1  # default value
        human2.dynamics.gamma = 10  # default value
        uh2 = uh2 + human2.dynamics.get_robot_control(human2.x, robot1.x)

        uh3 = human3.dynamics.get_goal_control(human3.x, human3.goal)  # without avoid robot control
        # human.dynamics.gamma = 1  # default value
        human3.dynamics.gamma = 10  # default value
        uh3 = uh3 + human3.dynamics.get_robot_control(human3.x, robot1.x)

        ur1 = ur_traj[:, [i]]

        xh1 = human1.step(uh1)
        xh2 = human2.step(uh2)
        xh3 = human3.step(uh3)
        xr1 = robot1.step(ur1)

        thres = 0.05 # risk tolerance

        if (np.abs((xh1[0] - xr1[0]) ** 2 + (xh1[2] - xr1[2]) ** 2) < sigma ** 2 and p1 > thres) or (np.abs(
                (xh2[0] - xr1[0]) ** 2 + (xh2[2] - xr1[2]) ** 2) < sigma ** 2 and p2 > thres) or (np.abs(
                (xh3[0] - xr1[0]) ** 2 + (xh3[2] - xr1[2]) ** 2) < sigma ** 2 and p3 > thres):
            safety = False
            # print('safety violated at step ' + str(i))

    distance = np.linalg.norm(xr1[[0, 2]] - robot.goal[[0, 2]])
    # distance = norm(xr1 - robot.goal)


    return safety, distance


def generate_trajectories(robot, human, belief_h, horizon, goals, r_goal, xh_hist, xr_hist, model, stats_file=None, stats=None, verbose = False, plot=False):
    # generate robot trajectory based on beliefs on human's goal
    # n_obs is number of synthetic obstacles
    # here we consider fixed number of goals n=3

    if stats is None:
        with open(stats_file, "rb") as f:
            stats = pickle.load(f)

    n_obs = 10
    obs = np.zeros([n_obs, 4])
    obs[:, 0] = robot.x[0]
    obs[:, 2] = robot.x[2]

    dx = np.zeros([n_obs-1, 1])
    dy = np.zeros([n_obs-1, 1])

    d_min = 10000

    for k in range(n_obs):
        # if k == 0:
        #     obs[k, 0] = robot.x[0] + (robot.goal[0] - robot.x[0]) / np.sqrt(
        #         (robot.goal[0] - robot.x[0]) ** 2 + (robot.goal[2] - robot.x[2]) ** 2) * 2
        #     obs[k, 2] = robot.x[2] + (robot.goal[2] - robot.x[2]) / np.sqrt(
        #         (robot.goal[0] - robot.x[0]) ** 2 + (robot.goal[2] - robot.x[2]) ** 2) * 2
        if 0 <= k < 3:
            dx = -1
        elif 3 <= k < 6:
            dx = 0
        else:
            dx = 1

        if k == 0 or k == 3 or k == 6:
            dy = -1
        elif k == 1 or k == 4 or k == 7:
            dy = 0
        elif k == 2 or k == 5 or k == 8:
            dy = 1

        if k == n_obs - 1:
            # no obstacle (obstacle very far away, just for easier visualization purpose)
            obs[k, 0] = 1000
            obs[k, 2] = 1000
        else:
            obs[k, 0] = robot.x[0] + (robot.goal[0] - robot.x[0]) / np.sqrt(
                (robot.goal[0] - robot.x[0]) ** 2 + (robot.goal[2] - robot.x[2]) ** 2) * 2 + dx
            obs[k, 2] = robot.x[2] + (robot.goal[2] - robot.x[2]) / np.sqrt(
                (robot.goal[0] - robot.x[0]) ** 2 + (robot.goal[2] - robot.x[2]) ** 2) * 2 + dy


        # safety = True

        x_r = robot.x
        x_h = human.x
        robot1 = copy.deepcopy(robot)
        human1 = copy.deepcopy(human)
        human2 = copy.deepcopy(human)
        human3 = copy.deepcopy(human)

        human1.goal = goals[:, [0]]
        human2.goal = goals[:, [1]]
        human3.goal = goals[:, [2]]
        robot1.goal = r_goal

        xr = x_r
        xh1 = x_h
        xh2 = x_h
        xh3 = x_h

        # print(belief_h) # check out format of NN belief output # [0.32720825 0.281003   0.3917887 ]

        p1 = belief_h[0]
        p2 = belief_h[1]
        p3 = belief_h[2]

        xh1_traj = np.zeros((human1.dynamics.n, horizon + 1))
        xh2_traj = np.zeros((human2.dynamics.n, horizon + 1))
        xh3_traj = np.zeros((human3.dynamics.n, horizon + 1))
        xr1_traj = np.zeros((robot1.dynamics.n, horizon + 1))
        ur1_traj = np.zeros((robot1.dynamics.m, horizon + 1))
        xh1_traj[:, [0]] = xh1
        xh2_traj[:, [0]] = xh2
        xh3_traj[:, [0]] = xh3
        xr1_traj[:, [0]] = xr

        d = 1  # safe distance between human and robot

        for i in range(horizon):
            if plot:
                # plotting
                ax.cla()
                # plot trajectory trail so far
                overlay_timesteps(ax, xh1_traj[:, 0:i], xr1_traj[:, 0:i], n_steps=i)

                # TODO: highlight robot's predicted goal of the human
                ax.scatter(obs[k, 0], obs[k, 2], c="gold", s=100)
                ax.scatter(goals[0], goals[2], c="green", s=100)
                ax.scatter(human1.x[0], human1.x[2], c="#034483", s=100)
                ax.scatter(human1.x[0], human1.x[2], c="#034483", s=100 + 50 * i, alpha=p1, edgecolors='none')
                ax.scatter(human2.x[0], human2.x[2], c="#034483", s=100)
                ax.scatter(human2.x[0], human2.x[2], c="#034483", s=100 + 50 * i, alpha=p2, edgecolors='none')
                ax.scatter(human3.x[0], human3.x[2], c="#034483", s=100)
                ax.scatter(human3.x[0], human3.x[2], c="#034483", s=100 + 50 * i, alpha=p3, edgecolors='none')
                ax.scatter(robot1.x[0], robot1.x[2], c="#800E0E", s=100)
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 15)
                img_path = f"./data/videos/mfi_demo/generate_traj_loop"
                img_path += f"/{i:03d}.png"
                plt.savefig(img_path, dpi=300)


            uh1 = human1.dynamics.get_goal_control(human1.x, human1.goal)  # without avoid robot control
            # human.dynamics.gamma = 1  # default value
            human1.dynamics.gamma = 10  # default value
            uh1 = uh1 + human1.dynamics.get_robot_control(human1.x,
                                                          robot1.x)  # potential field control to avoid robot, parameter gamma to tune the aggressiveness

            uh2 = human2.dynamics.get_goal_control(human2.x, human2.goal)  # without avoid robot control
            # human.dynamics.gamma = 1  # default value
            human2.dynamics.gamma = 10  # default value
            uh2 = uh2 + human2.dynamics.get_robot_control(human2.x, robot1.x)

            uh3 = human3.dynamics.get_goal_control(human3.x, human3.goal)  # without avoid robot control
            # human.dynamics.gamma = 1  # default value
            human3.dynamics.gamma = 10  # default value
            uh3 = uh3 + human3.dynamics.get_robot_control(human3.x, robot1.x)

            # uh = human.get_u(robot.x)  # with avoid robot control. this will change human's goal as well
            ur1 = robot1.dynamics.get_goal_control(robot1.x, robot1.goal)
            # print('ur: ')
            # print(ur1)
            robot1.dynamics.gamma = 10  # default value
            ur1 = ur1 + robot1.dynamics.get_potential_field_control(robot1.x, obs[k])
            # avoid both obstacle and human
            sigma0 = 0.1  # initial std for uncertainty
            sigma = np.sqrt(i) * sigma0
            ur1 = robot1.dynamics.get_goal_control(robot1.x, robot1.goal) + robot1.dynamics.get_potential_field_control(
                robot1.x, obs[k].reshape(-1, 1)) + p1 * robot1.dynamics.get_potential_field_control(robot1.x, human1.x,
                                                                                                   sigma) + p2 * robot1.dynamics.get_potential_field_control(
                robot1.x, human2.x, sigma) + p3 * robot1.dynamics.get_potential_field_control(robot1.x, human3.x, sigma)

            if k == n_obs - 1:
                ur1 = robot1.dynamics.get_goal_control(robot1.x,
                                                       robot1.goal) + p1 * robot1.dynamics.get_potential_field_control(robot1.x,
                                                                                                       human1.x,
                                                                                                       sigma) + p2 * robot1.dynamics.get_potential_field_control(
                    robot1.x, human2.x, sigma) + p3 * robot1.dynamics.get_potential_field_control(robot1.x, human3.x,
                                                                                                 sigma)

            # if i > horizon - 30:  # only use goal control after horizon-30 time steps to remove steady state error
            #     ur1 = (30 / (horizon - i)) * robot1.dynamics.get_goal_control(robot1.x, robot1.goal)

            xh1 = human1.step(uh1)
            xh2 = human2.step(uh2)
            xh3 = human3.step(uh3)
            xr1 = robot1.step(ur1)

            xh1_traj[:, [i + 1]] = xh1
            xh2_traj[:, [i + 1]] = xh2
            xh3_traj[:, [i + 1]] = xh3
            xr1_traj[:, [i + 1]] = xr1
            ur1_traj[:, [i]] = ur1

            # don't need to check safety here, will feed back to NN to check safety

            # if np.abs((xh1[0] - xr1[0]) ** 2 + (xh1[2] - xr1[2]) ** 2) < sigma ** 2 or np.abs(
            #         (xh2[0] - xr1[0]) ** 2 + (xh2[2] - xr1[2]) ** 2) < sigma ** 2 or np.abs(
            #         (xh3[0] - xr1[0]) ** 2 + (xh3[2] - xr1[2]) ** 2) < sigma ** 2:
            #     safety = False
            #     if verbose:
            #         print('safety violated at step ' + str(i) + ' for obs ' + str(k))


        k_hist = 5
        k_plan = 20

        softmax = torch.nn.Softmax(dim=1)

        feats = True

        # single robot goal
        goal = r_goal
        # xr_plan = get_robot_plan(robot, horizon=k_plan, goal=goal)
        # print(xr_plan)
        xr_plan = xr1_traj[:, 0:k_plan]
        # print(xr_plan)
        if feats:
            hist_feats, future_feats = compute_features(xh_hist, xr_hist, xr_plan, goals)
            input_hist, input_future, input_goals = process_model_input(xh_hist, xr_hist, xr_plan.T, goals)
            input_hist = torch.cat((input_hist, torch.tensor(hist_feats.T).float().unsqueeze(0)), dim=2)
            input_future = torch.cat((input_future, torch.tensor(future_feats.T).float().unsqueeze(0)), dim=2)
            # normalize features with same mean and std as training data
            if stats is not None:
                input_hist = (input_hist - stats["input_traj_mean"]) / stats["input_traj_std"]
                input_future = (input_future - stats["robot_future_mean"]) / stats["robot_future_std"]
                input_goals = (input_goals.transpose(1, 2) - stats["input_goals_mean"]) / stats[
                    "input_goals_std"]
                input_goals = input_goals.transpose(1, 2)
            model_out = model(input_hist, input_future, input_goals)
        else:
            model_out = model(*process_model_input(xh_hist, xr_hist, xr_plan.T, goals))
        belief_h = softmax(model_out).detach().numpy()[0]

        # feed trajectory back to NN to examine safety
        safety, distance = check_safety(robot, human, belief_h, horizon, goals, r_goal, ur1_traj)

        if verbose:
            print('safety for obs' + str(k) )
            print(safety)
            print('distance for obs' + str(k))
            print(distance)

        # distance = norm(xr1 - robot.goal)

        if safety == True and distance < d_min:
            d_min = distance
            safe_traj = ur1_traj
            obs_loc = obs[k,:] # for visualization

    # safety + efficiency
    # in outer loop: + human_goal (belief) + robot_goal to form the final cost function

    return safety, safe_traj, distance, belief_h, obs_loc



if __name__ == "__main__":
    horizon = 80
    ts = 0.05

    if horizon is None:
        fixed_horizon = False
        horizon = np.inf
        arr_size = 100
    else:
        fixed_horizon = True
        arr_size = horizon

    # xr0 = np.array([[1.0, 0.0, -1.0, 0.0]]).T
    # xh0 = np.array([[-4.0, 0.0, 0.0, 0.0]]).T

    xh0 = np.random.uniform(-10, 10, (4, 1))
    xh0[[1, 3]] = 0
    xr0 = np.random.uniform(-10, 10, (4, 1))
    xr0[[1, 3]] = 0

    # random goal locations
    goals = np.random.uniform(-10, 10, (4, 3))
    goals[[1, 3]] = 0
    r_goal = goals[:, [2]]

    h_dynamics = DIDynamics(ts=ts)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.0005)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=5)
    r_belief = BayesEstimator(goals, h_dynamics, beta=0.0005) # standard Bayesian estimator

    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    robot.set_goals(goals)

    robot.goal = goals[:, [2]]
    human.goal = goals[:, [0]]


    robot.x = xr0
    human.x = xh0

    num_goal = 3
    beliefs_h = np.zeros([horizon, num_goal])  # record beliefs on human
    beliefs_r = np.zeros([horizon, num_goal])  # record beliefs on robot
    goals_r = np.zeros([horizon, 1])  # record robot's goal

    # model_path = "./data/models/prob_pred_intention_predictor_bayes_20230602-210158.pt"
    # model_path = "./data/models/prob_pred_intention_predictor_bayes_20230623-12.pt"
    model_path = "./data/models/prob_pred_intention_predictor_bayes_20230804-073911.pt"
    # stats_file = "./data/models/prob_pred_intention_predictor_bayes_20230623-12_stats.pkl"
    stats_file = "./data/models/bayes_prob_branching_processed_feats_stats.pkl"

    nn_horizon = 20
    hidden_size = 128
    num_layers = 2
    hist_feats = 21
    plan_feats = 10
    model = create_model(horizon_len=nn_horizon, hidden_size=hidden_size, num_layers=num_layers, hist_feats=hist_feats,
                         plan_feats=plan_feats)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    xh_traj = np.zeros((human.dynamics.n, arr_size + 1))
    xr_traj = np.zeros((robot.dynamics.n, arr_size + 1))
    xh_traj[:, [0]] = xh0
    xr_traj[:, [0]] = xr0

    k_hist = 5
    k_plan = 20

    softmax = torch.nn.Softmax(dim=1)

    feats = True

    fig, ax = plt.subplots()
    for i in range(horizon):
        startTimer = datetime.datetime.now()

        if stats_file is not None:
            with open(stats_file, "rb") as f:
                stats = pickle.load(f)
        else:
            stats = None

        ## new model
        if i < k_hist:
            # get zero-padded history of both agents
            xh_hist = np.hstack((np.zeros((human.dynamics.n, k_hist - i)), xh_traj[:, 0:i]))
            xr_hist = np.hstack((np.zeros((robot.dynamics.n, k_hist - i)), xr_traj[:, 0:i]))
        else:
            xh_hist = xh_traj[:, i - k_hist:i]
            xr_hist = xr_traj[:, i - k_hist:i]

        # original goal of robot
        goal = robot.goal

        # Bayesian belief
        belief_h = r_belief.belief

        min_cost = 10000

        # choose over multiple robot goals
        for goal_idx in range(robot.goals.shape[1]):
            r_goal = robot.goals[:, [goal_idx]]

            ## TODO: form loop over all possible r_goal
            safety, ur_traj, distance, new_belief, obs_loc = generate_trajectories(robot, human, belief_h, nn_horizon, goals, r_goal, xh_hist, xr_hist,
                                                    model, stats_file, plot=False)

            # print('safety for goal' + str(goal_idx))
            # print(safety)
            # print('distance for goal' + str(goal_idx))
            # print(distance)
            # print('belief difference for goal' + str(goal_idx))
            # print(belief_h - new_belief)

            # choose trajectory based on the cost function
            cost = 0
            if safety == False:
                cost = cost + 10
            # cost += distance + np.linalg.norm(belief_h - new_belief, axis=0)
            cost += distance
            if not np.array_equal(r_goal, goal):
                # print('changed goal')
                cost += 5

            if cost < min_cost:
                min_cost = cost
                ur_traj_selected = ur_traj
                robot.goal = r_goal
                obs = obs_loc


        uh = human.get_u(robot.x) # include get_goal_control, get_robot_control, get_goal

        ur = ur_traj_selected[:, [0]]

        r_belief.update_belief(human.x, uh)
        # update human's belief
        human.update_belief(robot.x, ur)

        xh = human.step(uh)
        xr = robot.step(ur)

        beliefs_h[i, :] = belief_h
        beliefs_r[i, :] = human.belief.belief

        goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

        ax.cla()
        ax.scatter(obs[0], obs[2], c="gold", s=100)
        ax.scatter(goals[0], goals[2], c=goal_colors, s=100)
        ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
        ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
        # ax.scatter(robot.goal[0], robot.goal[2], c="red", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 15)
        # img_path = f"./data/videos/mfi_demo/traj_loop_nn_multi"
        # img_path += f"/{i:03d}.png"
        # plt.savefig(img_path, dpi=300)
        plt.pause(0.01)

    plt.figure()
    plt.plot(beliefs_h[:, 0], label="P(g0)", c=goal_colors[0])
    plt.plot(beliefs_h[:, 1], label="P(g1)", c=goal_colors[1])
    plt.plot(beliefs_h[:, 2], label="P(g2)", c=goal_colors[2])
    plt.legend(['goal 1', 'goal 2', 'goal 3'])
    # plt.savefig(f"./data/videos/mfi_demo/traj_loop_nn_multi/beliefs_h", dpi=300)
    plt.pause(0.01)

    plt.figure()
    plt.plot(beliefs_r[:, 0], label="P(g0)", c=goal_colors[0])
    plt.plot(beliefs_r[:, 1], label="P(g1)", c=goal_colors[1])
    plt.plot(beliefs_r[:, 2], label="P(g2)", c=goal_colors[2])
    plt.legend(['goal 1', 'goal 2', 'goal 3'])
    # plt.savefig(f"./data/videos/mfi_demo/traj_loop_nn_multi/beliefs_r", dpi=300)

    plt.show()




