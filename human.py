import numpy as np

from agent import BaseAgent
from dynamics import Dynamics

class Human(BaseAgent):
    def __init__(self, x0, dynamics : Dynamics, goals, goal=None, gamma=1):
        self.x = x0
        self.dynamics = dynamics
        self.dynamics.gamma = gamma
        self.goals = goals
        self.gamma = gamma

        # TODO: change this to use an index, or make sure the goal is in the set of goals
        if goal is not None:
            self.set_goal = True
            self.goal = goal
        else:
            self.set_goal = False

        # TODO: set control limits

    def get_goal(self, get_idx=False):
        # if goal was set on initialization, return that
        if self.set_goal:
            return self.goal

        # otherwise, current goal will be the closest by Euclidean distance
        goal_xy = self.goals[[0,2],:]
        dists = np.linalg.norm(goal_xy-self.x[[0,2]], axis=0)

        min_i = np.argmin(dists)

        goal = self.goals[:,[min_i]]
        if get_idx:
            return goal, min_i
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

    def copy(self):
        if self.set_goal:
            return Human(self.x, self.dynamics, self.goals, self.goal)
        return Human(self.x, self.dynamics, self.goals)
