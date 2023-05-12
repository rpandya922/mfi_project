import numpy as np
import control

def integrate_rk4(model, x0, u, t0, ts):
    """
    x0: initial state
    u: control
    t0: initial time
    ts: sampling time
    """
    k1 = ts * model.dx(x0, u, t0)
    k2 = ts * model.dx(x0 + 0.5 * k1, u, t0 + 0.5 * ts)
    k3 = ts * model.dx(x0 + 0.5 * k2, u, t0 + 0.5 * ts)
    k4 = ts * model.dx(x0 + k3, u, t0 + ts)

    return x0 + ((k1 + 2*(k2 + k3) + k4) / 6.0)

class Unicycle():
    def __init__(self, ts : float, W : np.ndarray = None):
        self.ts = ts # sampling time
        if W is None:
            W  = np.zeros((4, 4))
        self.W = W

        self.kv = 1.7
        self.kpsi = 1

    def dx(self, x, u, t):
        """
        x: [x, y, v, psi]
        u: [v_dot, psi_dot]
        t: time (unused because dynamics are time-invariant)
        """

        x_dot = np.zeros((4, 1))
        x_dot[0] = x[2] * np.cos(x[3])
        x_dot[1] = x[2] * np.sin(x[3])
        x_dot[2] = u[0]
        x_dot[3] = u[1]

        return x_dot + np.random.multivariate_normal(np.zeros(4), self.W, size=(1,)).T
    
    def compute_goal_control(self, x, goal):
        """
        x: [x, y, v, phi]
        goal: [x, y]
        """
        v_dot = -((x[0] - goal[0])*np.cos(x[3]) + (x[1] - goal[1])*np.sin(x[3])) - (self.kv * x[2])
        psi_dot = self.kpsi * (np.arctan((x[1] - goal[1])/(x[0] - goal[0]) - x[3]))

        return np.array([v_dot, psi_dot])

    def step(self, x, u):
        return integrate_rk4(self, x, u, 0, self.ts)

class LTI():
    def __init__(self, ts : float, W : np.ndarray = None, gamma = 1):
        self.ts = ts # sampling time
        self.gamma = gamma # weight for avoidance control

        if W is None:
            W  = np.zeros((4, 4))
        self.W = W

        self.n = 4
        self.m = 2

        self.A = np.zeros((4, 4))
        self.A[0,1] = 1
        self.A[2,3] = 1
        self.B = np.zeros((4, 2))
        self.B[1,0] = 1
        self.B[3,1] = 1

        # compute LQR gain for LTI system
        K, _, _ = control.lqr(self.A, self.B, np.eye(self.n), np.eye(self.m))
        self.K = K
        self.K2 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    
    def dx(self, x, u, t):
        return self.A @ x + self.B @ u + np.random.multivariate_normal(np.zeros(self.n), self.W, size=(1,)).T

    def compute_control(self, x, goal, obstacles = None):
        """
        x: [x, x_dot, y, y_dot]
        goal: [x, x_dot, y, y_dot]
        """
        
        uh = -self.K @ (x - goal)

        if obstacles is not None:
            # obstacles are given only as x,y coordinates
            # compute distance to each obstacle
            dist = np.linalg.norm(obstacles - x[[0,2]], axis=0)
            # compute avoidance control
            ua = np.zeros((2,1))
            for i, d in enumerate(dist):
                # make sure d doesn't get to 0 (for numerical stability)
                d = max(d, 0.1)
                obs = np.array([[obstacles[0, i], 0, obstacles[1, i], 0]]).T
                ua += self.K2 @ (x - obs) * self.gamma / d**2
            uh += ua
        
        # return total control
        return uh
    
    def step(self, x, u):
        return integrate_rk4(self, x, u, 0, self.ts)