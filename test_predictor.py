import numpy as np
import torch
softmax = torch.nn.Softmax(dim=1)
import matplotlib.pyplot as plt

from human import Human
from robot import Robot
from dynamics import DIDynamics
from intention_predictor import IntentionPredictor, create_model

def get_robot_plan(robot, horizon=5, return_controls=False):
    # ignore safe control for plan
    robot_x = robot.x
    robot_states = np.zeros((robot.dynamics.n, horizon))
    robot_controls = np.zeros((robot.dynamics.m, horizon))
    for i in range(horizon):
        goal_u = robot.dynamics.get_goal_control(robot_x, robot.goal)
        robot_x = robot.dynamics.step(robot_x, goal_u)
        robot_states[:,[i]] = robot_x
        robot_controls[:,[i]] = goal_u

    if return_controls:
        return robot_states, robot_controls
    return robot_states

def get_empty_robot_plan(robot, horizon=5):
    robot_states = np.hstack([robot.x for i in range(horizon)])

    return robot_states

if __name__ == "__main__":
    model = create_model()
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor.pt"))
    plot_sim = False

    ts = 0.05
    horizon = 100

    # np.random.seed(0)
    # randomly initialize xh0, xr0, goals
    xh0 = np.random.uniform(size=(4, 1))*20 - 10
    xh0[[1,3]] = np.zeros((2, 1))
    xr0 = np.random.uniform(size=(4, 1))*20 - 10
    xr0[[1,3]] = np.zeros((2, 1))

    goals = np.random.uniform(size=(4, 3))*20 - 10
    goals[[1,3],:] = np.zeros((2, 3))
    r_goal = goals[:,[np.random.randint(0,3)]]

    dynamics_h = DIDynamics(ts)
    human = Human(xh0, dynamics_h, goals)
    dynamics_r = DIDynamics(ts)
    robot = Robot(xr0, dynamics_r, r_goal)

    xh_traj = np.zeros((4, horizon))
    xr_traj = np.zeros((4, horizon))
    h_goals = np.zeros((4, horizon))
    h_goal_reached = np.zeros((1, horizon))

    if plot_sim:
        fig, ax = plt.subplots()

    for i in range(horizon):
        # plot human, robot, and goals
        if plot_sim:
            ax.cla()
            ax.scatter(human.x[0], human.x[2])
            ax.scatter(robot.x[0], robot.x[2])
            ax.scatter(goals[0], goals[2])
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            plt.pause(0.01)

        # save data
        xh_traj[:,[i]] = human.x
        xr_traj[:,[i]] = robot.x
        h_goals[:,[i]] = human.get_goal()
        # check if human reached its goal
        if np.linalg.norm(human.x - human.get_goal()) < 0.1:
            h_goal_reached[:,i] = 1

        # take step
        uh = human.get_u(robot.x)
        if i == 0:
            ur = robot.get_u(human.x, robot.x, human.x)
        else:
            ur = robot.get_u(human.x, xr_traj[:,[i-1]], xh_traj[:,[i-1]])

        xh = human.step(uh)
        xr = robot.step(ur)

        if i > 5:
            # building the inputs to the model
            xh_hist = xh_traj[:,i-5:i]
            xr_hist = xr_traj[:,i-5:i]
            traj_hist = torch.tensor(np.vstack((xh_hist, xr_hist)).T).float().unsqueeze(0)
            r_plan = get_robot_plan(robot)
            r_plan = torch.tensor(r_plan.T).float().unsqueeze(0)
            input_goals = torch.tensor(goals).float().unsqueeze(0)

            goal_probs = softmax(model(traj_hist, r_plan, input_goals))
            est_goal_idx = torch.argmax(goal_probs).item()

            _, h_goal_idx = human.get_goal(get_idx=True)

            print(h_goal_idx, est_goal_idx, goal_probs[0,est_goal_idx].item())

