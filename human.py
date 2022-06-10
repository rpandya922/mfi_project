from abc import ABC, abstractmethod
import numpy as np
import control
import ipdb

class Dynamics(ABC):
    @abstractmethod
    def step(self, x, u):
        pass

    @abstractmethod
    def get_goal_control(self, x, goal):
        pass

class Human():
    def __init__(self, x0, dynamics : Dynamics, goal):
        self.x = x0
        self.dynamics = dynamics
        self.goal = goal
        # TODO: set control limits

        # TODO: have robot position passed in to controller
        self.robot_pos = np.array([[6.0, 0.0, 5.0, 0.0]]).T

    def set_goal(self, goal):
        self.goal = goal

    def get_u(self):
        # get control that moves human towards goal
        goal_u = self.dynamics.get_goal_control(self.x, self.goal)
        # get control that pushes human away from robot
        robot_avoid_u = self.dynamics.get_robot_control(self.x, self.robot_pos)
        return goal_u + robot_avoid_u

    def step(self, u):
        self.x = self.dynamics.step(self.x, u)
        return self.x

class DIDynamics(Dynamics):
    """
    Double integrator linear dynamics
    """
    def __init__(self, ts: float):
        """
        ts : sampling time
        """
        self.ts = ts
        self.A = np.array([[1, self.ts, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, self.ts],
                           [0, 0, 0, 1]])
        self.B = np.array([[0.5*(self.ts**2), 0],
                           [self.ts, 0],
                           [0, 0.5*(self.ts**2)],
                           [0, self.ts]])

        self.n = 4
        self.m = 2

        # save discrete-time LQR control
        K, _, _ = control.dlqr(self.A, self.B, 10*np.eye(self.n), np.eye(self.m))
        self.K = K

        self.gamma = 10


    def step(self, x, u):
        return self.A @ x + self.B @ u

    def get_goal_control(self, x, goal):
        # return LQR control
        u = -self.K @ (x - goal)

        return u

    def get_robot_control(self, x, xr):
        # use LQR gain to get robot avoidance control
        d = np.linalg.norm(x[[0,2]] - xr[[0,2]])
        u = (self.gamma / d**2) * self.K @ (x - xr)

        return u
