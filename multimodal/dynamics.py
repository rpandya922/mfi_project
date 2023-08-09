import numpy as np
import control

def integrate_rk4(dx, x0, u, t0, ts):
    """
    x0: initial state
    u: control
    t0: initial time
    ts: sampling time
    """
    k1 = ts * dx(x0, u, t0)
    k2 = ts * dx(x0 + 0.5 * k1, u, t0 + 0.5 * ts)
    k3 = ts * dx(x0 + 0.5 * k2, u, t0 + 0.5 * ts)
    k4 = ts * dx(x0 + k3, u, t0 + ts)

    return x0 + ((k1 + 2*(k2 + k3) + k4) / 6.0)

class Unicycle():
    def __init__(self, ts : float, W : np.ndarray = None, kv = 1.7, kpsi = 1):
        self.ts = ts # sampling time
        if W is None:
            W  = np.zeros((4, 4))
        self.W = W

        self.kv = kv
        self.kpsi = kpsi

    def f(self, x):
        return np.array([x[2] * np.cos(x[3]), x[2] * np.sin(x[3]), [0], [0]])
    
    def g(self, x):
        return np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

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
        # compute angle to goal
        x_diff = goal[0] - x[0]
        y_diff = goal[1] - x[1]
        # compute angle between goal and velocity
        vel_angle = x[3]
        goal_angle = np.arctan(y_diff/x_diff)
        angle = goal_angle - vel_angle

        if angle > np.pi/2:
            goal_angle = goal_angle - np.pi
        elif angle < -np.pi/2:
            goal_angle = goal_angle + np.pi

        # trying a 2-stage controller that first moves angle to be <90deg
        angle2 = goal_angle - vel_angle
        if abs(angle2) > np.pi/2:
            v_dot = 0.0*x[2]
        else:
            v_dot = -((x[0] - goal[0])*np.cos(x[3]) + (x[1] - goal[1])*np.sin(x[3])) - (self.kv * x[2])

        psi_dot = self.kpsi*(goal_angle - vel_angle)
        # psi_dot = self.kpsi * (np.arctan((x[1] - goal[1])/(x[0] - goal[0])) - x[3])

        return np.array([v_dot, psi_dot])

    def step(self, x, u):
        return integrate_rk4(self.dx, x, u, 0, self.ts)

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
        self.Q = 5*np.eye(self.n)
        self.R = np.eye(self.m)
        K, P, _ = control.lqr(self.A, self.B, self.Q, self.R)
        self.K = K
        self.P = P # solution to Riccati equation (use for computing Q function)
        
        Ad = np.array([[1, self.ts, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, self.ts],
                           [0, 0, 0, 1]])
        Bd = np.array([[0.5*(self.ts**2), 0],
                           [self.ts, 0],
                           [0, 0.5*(self.ts**2)],
                           [0, self.ts]])
        Kd, Pd, _ = control.dlqr(Ad, Bd, self.Q, self.R)
        self.Kd = Kd
        self.Pd = Pd # solution to discrete-time Riccati equation (use for computing Q function)

        self.K2 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    
    def mean_dyn(self, x, u, t):
        return self.A @ x + self.B @ u

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
        return integrate_rk4(self.dx, x, u, 0, self.ts)

    def step_mean(self, x, u):
        return integrate_rk4(self.mean_dyn, x, u, 0, self.ts)
