from abc import ABC, abstractmethod
import numpy as np
import control

class Dynamics(ABC):
    @abstractmethod
    def step(self, x, u):
        pass

    @abstractmethod
    def get_goal_control(self, x, goal):
        pass

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
        self.K2 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        self.gamma = 1


    def step(self, x, u):
        return self.A @ x + self.B @ u

    def get_goal_control(self, x, goal):
        # return LQR control
        u = -self.K @ (x - goal)

        return u

    # TODO: rename (and/or move to human.py?)
    def get_robot_control(self, x, xr):
        # use LQR gain to get robot avoidance control
        d = np.linalg.norm(x[[0,2]] - xr[[0,2]])

        # make sure d doesn't get to 0 (for numerical stability)
        d = max(d, 0.1)
        # if d > 5:
        #     return np.zeros((self.m, 1))
        u = (self.gamma / d**2) * self.K2 @ (x - xr)

        return u