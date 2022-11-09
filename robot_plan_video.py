import numpy as np
import matplotlib.pyplot as plt
import torch
softmax = torch.nn.Softmax(dim=1)

from intention_predictor import create_model
from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_utils import overlay_timesteps, get_robot_plan, process_model_input

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def isolated_human(human, robot, goals, horizon):
    # show the human's trajectory when the robot is not acting
    fig, ax = plt.subplots()

    xh = human.x
    xh_traj = np.zeros((human.dynamics.n, horizon+1))
    xh_traj[:,[0]] = xh
    for i in range(horizon):
        # plotting
        ax.cla()
        # plot trajectory trail so far
        overlay_timesteps(ax, xh_traj[:,0:i], [], n_steps=i)

        # TODO: highlight robot's predicted goal of the human
        ax.scatter(goals[0], goals[2], c="green", s=100)
        ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
        ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        img_path = f"./data/videos/mfi_demo/frames1"
        img_path += f"/{i:03d}.png"
        plt.savefig(img_path, dpi=300)
        # plt.pause(0.01)

        # compute agent controls
        uh = human.get_u(robot.x)

        # compute new states
        xh = human.step(uh)

        # save data
        xh_traj[:,[i+1]] = xh

def naive_robot(human, robot, goals, horizon):
    # show both agent trajectories when the robot doesn't reason about its effect on the human
    fig, ax = plt.subplots()

    xh = human.x
    xr = robot.x
    xh_traj = np.zeros((human.dynamics.n, horizon+1))
    xr_traj = np.zeros((robot.dynamics.n, horizon+1))
    xh_traj[:,[0]] = xh
    xr_traj[:,[0]] = xr
    for i in range(horizon):
        # plotting
        ax.cla()
        # plot trajectory trail so far
        overlay_timesteps(ax, xh_traj[:,0:i], xr_traj[:,0:i], n_steps=i)

        # TODO: highlight robot's predicted goal of the human
        ax.scatter(goals[0], goals[2], c="green", s=100)
        ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
        ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        img_path = f"./data/videos/mfi_demo/frames2"
        img_path += f"/{i:03d}.png"
        plt.savefig(img_path, dpi=300)
        # plt.pause(0.01)

        # compute the robot's prediction of the human's intention
        if i < 5:
            nominal_goal_idx = 1
        else:
            xh_hist = xh_traj[:,i-5:i]
            xr_hist = xr_traj[:,i-5:i]
            r_beliefs = []
            for goal_idx in range(robot.goals.shape[1]):
                goal = robot.goals[:,[goal_idx]]
                # compute robot plan given this goal
                # xr_plan = get_robot_plan(robot, horizon=20, goal=goal)
                xr_plan = np.tile(robot.x, 20)

                r_beliefs.append(softmax(model(*process_model_input(xh_hist, xr_hist, xr_plan.T, goals))).detach().numpy()[0])
            # use robot's belief to pick which goal to move toward by picking the one that puts the highest probability on the human's nominal goal (what we observe in the first 5 timesteps)
            r_beliefs = np.array(r_beliefs)
            r_goal_idx = np.argmax(r_beliefs[:,nominal_goal_idx])
            robot.set_goal(robot.goals[:,[r_goal_idx]])

        # compute agent controls
        uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
        # robot should stay still for 5 timesteps
        if i < 5:
            ur = np.zeros(ur.shape)

        # update human belief (if applicable)
        if type(human) == BayesHuman and i >=5:
            human.update_belief(robot.x, ur)

        # compute new states
        xh = human.step(uh)
        xr = robot.step(ur)

        # save data
        xh_traj[:,[i+1]] = xh
        xr_traj[:,[i+1]] = xr

def robot_with_intention(human, robot, goals, horizon, model):
    # show both agent trajectories when the robot reasons about its effect on the human
    fig, ax = plt.subplots()

    xh = human.x
    xr = robot.x

    xh_traj = np.zeros((human.dynamics.n, horizon+1))
    xr_traj = np.zeros((robot.dynamics.n, horizon+1))
    xh_traj[:,[0]] = xh
    xr_traj[:,[0]] = xr
    for i in range(horizon):
        # plotting
        ax.cla()
        # plot trajectory trail so far
        overlay_timesteps(ax, xh_traj[:,0:i], xr_traj[:,0:i], n_steps=i)

        # TODO: highlight robot's predicted goal of the human
        ax.scatter(goals[0], goals[2], c="green", s=100)
        ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
        ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        img_path = f"./data/videos/mfi_demo/frames3"
        img_path += f"/{i:03d}.png"
        plt.savefig(img_path, dpi=300)
        # plt.pause(0.01)

        # compute the robot's prediction of the human's intention
        if i < 5:
            nominal_goal_idx = 1
        else:
            xh_hist = xh_traj[:,i-5:i]
            xr_hist = xr_traj[:,i-5:i]
            r_beliefs = []
            for goal_idx in range(robot.goals.shape[1]):
                goal = robot.goals[:,[goal_idx]]
                # compute robot plan given this goal
                xr_plan = get_robot_plan(robot, horizon=20, goal=goal)

                r_beliefs.append(softmax(model(*process_model_input(xh_hist, xr_hist, xr_plan.T, goals))).detach().numpy()[0])
            # use robot's belief to pick which goal to move toward by picking the one that puts the highest probability on the human's nominal goal (what we observe in the first 5 timesteps)
            r_beliefs = np.array(r_beliefs)
            r_goal_idx = np.argmax(r_beliefs[:,nominal_goal_idx])
            robot.set_goal(robot.goals[:,[r_goal_idx]])

        # compute agent controls
        uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
        # robot should stay still for 5 timesteps
        if i < 5:
            ur = np.zeros(ur.shape)

        # update human belief (if applicable)
        if type(human) == BayesHuman and i >=5:
            human.update_belief(robot.x, ur)

        # compute new states
        xh = human.step(uh)
        xr = robot.step(ur)

        # save data
        xh_traj[:,[i+1]] = xh
        xr_traj[:,[i+1]] = xr

if __name__ == "__main__":
    horizon = 100
    ts = 0.05
    k_plan = 20

    model = create_model(horizon_len=k_plan)
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor_bayes_ll.pt", map_location=device))
    model.eval()

    xh0 = np.array([[1.0, 0.0, -1.0, 0.0]]).T
    xr0 = np.array([[-4.0, 0.0, 0.0, 0.0]]).T

    goals = np.array([
        [1.0, 0.0, 6.0, 0.0],
        [2.0, 0.0, 4.0, 0.0],
        [-8.0, 0.0, 8.0, 0.0]
    ]).T
    r_goal = goals[:,[0]]

    h_dynamics = DIDynamics(ts=ts)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=0)

    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    robot.set_goals(goals)

    # isolated_human(human, robot, goals, horizon)
    # naive_robot(human, robot, goals, horizon)
    robot_with_intention(human, robot, goals, horizon, model)
