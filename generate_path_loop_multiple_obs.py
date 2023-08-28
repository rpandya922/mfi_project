# examine safety given human's goal and initial position and robot's goal and initial position
# assume both agents run nominal proportional controller

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
from numpy.linalg import norm
from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_predictor import create_model
# from intention_utils import initialize_problem, overlay_timesteps, get_robot_plan
from intention_utils import overlay_timesteps
# from examine_safety import check_safety
from cbp_model import *
import math
import copy
import datetime


def generate_trajectory_belief(robot, human, belief_h, horizon, goals, plot=False):
    # generate robot trajectory based on beliefs on human's goal (belief_h)
    # n_obs is number of synthetic obstacles
    # here we consider fixed number of goals n=3

    n_obs = 10
    obs = np.zeros([n_obs, 4])
    obs[:, 0] = robot.x[0]
    obs[:, 2] = robot.x[2]

    dx = np.zeros([n_obs-1, 1])
    dy = np.zeros([n_obs-1, 1])

    d_min = 10000

    for k in range(n_obs):
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
            # no obstacle (obstacle very far away for plotting purposes)
            obs[k, 0] = 1000
            obs[k, 2] = 1000
        else:
            # in between robot and its goal, with (dx, dy) perturbation
            obs[k, 0] = robot.x[0] + (robot.goal[0] - robot.x[0]) / np.sqrt(
                (robot.goal[0] - robot.x[0]) ** 2 + (robot.goal[2] - robot.x[2]) ** 2) * 2 + dx
            obs[k, 2] = robot.x[2] + (robot.goal[2] - robot.x[2]) / np.sqrt(
                (robot.goal[0] - robot.x[0]) ** 2 + (robot.goal[2] - robot.x[2]) ** 2) * 2 + dy


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

        # print('human2.goal: ')
        # print(human2.goal)

        xr = x_r
        xh1 = x_h
        xh2 = x_h
        xh3 = x_h

        p1 = belief_h.belief[0]
        p2 = belief_h.belief[1]
        p3 = belief_h.belief[2]

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
                # img_path = f"./data/videos/mfi_demo/generate_traj_loop"
                # img_path += f"/{i:03d}.png"
                # plt.savefig(img_path, dpi=300)


            uh1 = human1.dynamics.get_goal_control(human1.x, human1.goal)  # without avoid robot control
            # human.dynamics.gamma = 1  # default value
            human1.dynamics.gamma = 10
            uh1 = uh1 + human1.dynamics.get_robot_control(human1.x,
                                                          robot1.x)  # potential field control to avoid robot, parameter gamma to tune the aggressiveness

            uh2 = human2.dynamics.get_goal_control(human2.x, human2.goal)  # without avoid robot control
            # human.dynamics.gamma = 1  # default value
            human2.dynamics.gamma = 10
            uh2 = uh2 + human2.dynamics.get_robot_control(human2.x, robot1.x)

            uh3 = human3.dynamics.get_goal_control(human3.x, human3.goal)  # without avoid robot control
            # human.dynamics.gamma = 1  # default value
            human3.dynamics.gamma = 10
            uh3 = uh3 + human3.dynamics.get_robot_control(human3.x, robot1.x)

            # uh = human.get_u(robot.x)  # with avoid robot control. but this will change human's goal

            ur1 = robot1.dynamics.get_goal_control(robot1.x, robot1.goal)
            robot1.dynamics.gamma = 10
            ur1 = ur1 + robot1.dynamics.get_potential_field_control(robot1.x, obs[k])
            # avoid both obstacle and human
            sigma0 = 0.1  # initial std for uncertainty
            sigma = np.sqrt(i) * sigma0 # growing uncertainty over time
            ur1 = robot1.dynamics.get_goal_control(robot1.x, robot1.goal) + robot1.dynamics.get_potential_field_control(
                robot1.x, obs[k].reshape(-1, 1)) + p1 * robot1.dynamics.get_potential_field_control(robot1.x, human1.x,
                                                                                                   sigma) + p2 * robot1.dynamics.get_potential_field_control(
                robot1.x, human2.x, sigma) + p3 * robot1.dynamics.get_potential_field_control(robot1.x, human3.x, sigma)

            if k == n_obs - 1:
                # no obstacle
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

            thres = 0.05 # risk tolerance

            if (np.abs((xh1[0] - xr1[0]) ** 2 + (xh1[2] - xr1[2]) ** 2) < sigma ** 2 and p1 > thres) or (np.abs(
                    (xh2[0] - xr1[0]) ** 2 + (xh2[2] - xr1[2]) ** 2) < sigma ** 2 and p2 > thres) or (np.abs(
                    (xh3[0] - xr1[0]) ** 2 + (xh3[2] - xr1[2]) ** 2) < sigma ** 2 and p3 > thres):
                safety = False
                # print('safety violated at step ' + str(i))

        distance = norm(xr1 - robot.goal)

        if safety == True and distance < d_min:
            d_min = distance
            safe_traj = ur1_traj
            obs_loc = obs[k, :]  # for visualization

    return safety, safe_traj, obs_loc


if __name__ == "__main__":
    horizon = 80
    ts = 0.1
    traj_horizon = 20

    xr0 = np.array([[1.0, 0.0, -1.0, 0.0]]).T
    xh0 = np.array([[-4.0, 0.0, 0.0, 0.0]]).T

    # random goal locations
    # TODO: make sure the goals are not too close to each other
    goals = np.random.uniform(-10, 10, (4, 3))
    goals[[1, 3]] = 0

    r_goal = goals[:, [2]] # random initial goal for robot

    h_dynamics = DIDynamics(ts=ts)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.0005)
    belief_h = BayesEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=0)
    r_belief = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)

    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    robot.set_goals(goals)

    # random goals
    robot.goal = goals[:, [2]]
    human.goal = goals[:, [0]]

    # print(belief_h.belief)

    robot.x = xr0
    human.x = xh0

    num_goal = 3
    beliefs_h = np.zeros([horizon, num_goal])  # record beliefs on human
    goals_r = np.zeros([horizon, 1])  # record robot's goal

    fig, ax = plt.subplots()

    for i in range(horizon):
        startTimer = datetime.datetime.now()
        safety, ur_traj, obs_loc = generate_trajectory_belief(robot, human, belief_h, traj_horizon, goals, plot=False)
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        solver_time = deltaTimer.total_seconds()
        print('time of trajectory generation at each time step ')
        print(solver_time)
        print(safety)
        print(robot.goal)
        print(human.goal)

        uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # update human's belief
        human.update_belief(robot.x, ur)
        # simulate robot nominal belief update
        r_belief_nominal.belief = r_belief_nominal.update_belief(human.x, uh)
        # update robot's belief with cbp
        # r_belief_prior = r_belief.update_belief(human.x, uh, robot.x)
        r_belief_prior = r_belief_nominal.belief

        if i > -1:
            # simulate human's next state
            state = human.dynamics.A @ human.x + human.dynamics.B @ uh
            # loop through goals and compute belief update for each
            divs = []
            posts = []
            for goal_idx in range(goals.shape[1]):
                goal = goals[:, [goal_idx]]
                # compute CBP belief update
                r_belief_post = r_belief.weight_by_score(r_belief_prior, goal, state, beta=0.5)
                posts.append(r_belief_post)
                divs.append(entropy(r_belief_post, r_belief_prior))
                # we don't want KL divergence, we want the one that puts the highest probability on human's most likely goal
            # pick the goal with the lowest divergence
            # goal_idx = np.argmin(divs)
            # pick the goal that puts highest probability on goal 0
            # goal_idx = np.argmax([p[0] for p in posts])
            # pick goal with highest probability on human's most likely goal
            goal_idx = np.argmax([p[np.argmax(r_belief_prior)] for p in posts])
            # goal_idx = np.argmax([p[h_goal_idx] for p in posts])
            # goal_idx = np.argmax([p[np.argmax(r_belief_nominal.belief)] for p in posts])
            # picks the goal that human is least likely to go towards
            # goal_idx = np.argmin([p[np.argmin(r_belief_prior)] for p in posts])
            robot.goal = goals[:, [goal_idx]]
            # update robot's belief
            r_belief_post = posts[goal_idx]
            r_belief.belief = r_belief_post
            # r_belief.belief = r_belief_prior
        else:
            r_belief.belief = r_belief_prior

        uh = human.dynamics.get_goal_control(human.x, human.goal)  # without avoid robot control
        uh = uh + human.dynamics.get_robot_control(human.x, robot.x)

        ur = ur_traj[:, [0]]

        xh = human.step(uh)
        xr = robot.step(ur)

        belief_h.update_belief(human.x, uh)

        beliefs_h[i, :] = belief_h.belief

        ax.cla()
        ax.scatter(obs_loc[0], obs_loc[2], c="gold", s=100)
        ax.scatter(goals[0], goals[2], c="green", s=100)
        ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
        ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 15)
        # img_path = f"./data/videos/mfi_demo/traj_loop_cbp_multi"
        # img_path += f"/{i:03d}.png"
        # plt.savefig(img_path, dpi=300)
        plt.pause(0.001)

    plt.figure()
    plt.plot(beliefs_h)
    plt.legend(['goal 1', 'goal 2', 'goal 3'])
    # plt.savefig(f"./data/videos/mfi_demo/traj_loop_cbp_multi/beliefs", dpi=300)
    plt.show()

    # safety = generate_trajectory_belief(robot, human, belief_h, horizon, goals, plot=True)
    # print(safety)



