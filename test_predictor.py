import numpy as np
import torch

from human import Human
from robot import Robot
from dynamics import DIDynamics
from intention_predictor import IntentionPredictor, create_model

def get_robot_plan(robot, horizon=5):
    # ignore safe control for plan
    robot_x = robot.x
    robot_states = np.zeros((robot.dynamics.n, horizon))
    for i in range(horizon):
        goal_u = robot.dynamics.get_goal_control(robot_x, robot.goal)
        robot_x = robot.dynamics.step(robot_x, goal_u)
        robot_states[:,[i]] = robot_x

    return robot_states

def get_empty_robot_plan(robot, horizon=5):
    robot_states = [robot.x for i in range(horizon)]

    return robot_states

if __name__ == "__main__":
    model = create_model()
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor.pt"))

    ts = 0.05
    horizon = 100

    np.random.seed(0)
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

    for i in range(horizon):
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
            r_plan = get_robot_plan(robot)
            import ipdb; ipdb.set_trace()

