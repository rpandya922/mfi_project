import numpy as np

from dynamics import Dynamics

import ipdb

class Human():
    def __init__(self, x0, dynamics : Dynamics, goals):
        self.x = x0
        self.dynamics = dynamics
        self.goals = goals

        # TODO: set control limits

    def get_goal(self):
        # current goal will be the closest by Euclidean distance
        goal_xy = self.goals[[0,2],:]
        try:
            dists = np.linalg.norm(goal_xy-self.x[[0,2]], axis=0)
        except:
            ipdb.set_trace()

        min_i = np.argmin(dists)

        goal = self.goals[:,[min_i]]
        return goal

    def set_goals(self, goals):
        self.goals = goals

    def get_u(self, robot_x):
        # get control that moves human towards goal
        goal = self.get_goal()
        goal_u = self.dynamics.get_goal_control(self.x, goal)
        # get control that pushes human away from robot
        robot_avoid_u = self.dynamics.get_robot_control(self.x, robot_x)
        return goal_u + robot_avoid_u

    def step(self, u):
        self.x = self.dynamics.step(self.x, u)
        return self.x
