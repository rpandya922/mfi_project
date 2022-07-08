from tkinter import E
import numpy as np

from agent import BaseAgent
from dynamics import Dynamics

class Robot(BaseAgent):
    def __init__(self, x0, dynamics : Dynamics, goal, dmin=3, eta=10, k_phi=0.1, lambda0=10):
        self.x = x0
        self.dynamics = dynamics
        # TODO: change this to take in a set of goals 
        self.goal = goal
        self.goals = np.array([goal])
        # TODO: set control limits

        # hyperparams for SSA
        self.dmin = dmin
        self.eta = eta
        self.k_phi = k_phi
        self.lambda0 = lambda0

    def get_goal(self):
        return self.goal

    def set_goal(self, goal):
        self.goal = goal

    def set_goals(self, goals):
        self.goals = goals

    def get_safe_control(self, xh, xr_prev, xh_prev, u_ref):
        """
        Safe Set Algorithm
        """

        xr = self.x
        # splitting human and robot state into position and velocity
        pr, vr = xr[[0, 2]].flatten(), xr[[1, 3]].flatten()
        ph, vh = xh[[0, 2]].flatten(), xh[[1, 3]].flatten()

        # COMPUTATION BASED ON BIS CODE
        p_M_p_X = np.zeros((6, 4))
        p_M_p_X[0,0] = 1
        p_M_p_X[1,1] = 1
        p_M_p_X[3,2] = 1
        p_M_p_X[4,3] = 1

        p_Mr_p_Xr = p_M_p_X.copy()
        p_Mh_p_Xh = p_M_p_X.copy()

        Mr = np.hstack((pr, [0], vr, [0]))
        Mh = np.hstack((ph, [0], vh, [0]))
        p_idx = [0, 1, 2]
        v_idx = [3, 4, 5]

        dot_Xr = (xr - xr_prev) / self.dynamics.ts 
        dot_Xh = (xh - xh_prev) / self.dynamics.ts
        
        dot_Xr = np.array([dot_Xr[0], dot_Xr[2], dot_Xr[1], dot_Xr[3]])
        dot_Xh = np.array([dot_Xh[0], dot_Xh[2], dot_Xh[1], dot_Xh[3]])
        fx = (self.dynamics.A - np.eye(4)) / self.dynamics.ts @ xr
        fu = self.dynamics.B / self.dynamics.ts
        fx = np.array([fx[0], fx[2], fx[1], fx[3]])
        fu = np.array([fu[0], fu[2], fu[1], fu[3]])

        d = np.linalg.norm(Mr[p_idx] - Mh[p_idx])

        dot_Mr = p_Mr_p_Xr @ dot_Xr
        dot_Mh = p_Mh_p_Xh @ dot_Xh

        dM = Mr - Mh
        dot_dM = dot_Mr - dot_Mh
        dp = dM[p_idx]
        dv = dM[v_idx]

        dot_dp = dot_dM[p_idx]
        dot_dv = dot_dM[v_idx]

        #dot_d is the component of velocity lies in the dp direction
        dot_d = dp.T @ dv / d

        p_dot_d_p_dp = dv / d - (dp.T @ dv) * dp / (d**3)
        p_dot_d_p_dv = dp / d
        
        p_dp_p_Mr = np.hstack([np.eye(3), np.zeros((3,3))])
        p_dp_p_Mh = -p_dp_p_Mr

        p_dv_p_Mr = np.hstack([np.zeros((3,3)), np.eye(3)])
        p_dv_p_Mh = -p_dv_p_Mr

        p_dot_d_p_Mr = p_dp_p_Mr.T @ p_dot_d_p_dp + p_dv_p_Mr.T @ p_dot_d_p_dv
        p_dot_d_p_Mh = p_dp_p_Mh.T @ p_dot_d_p_dp + p_dv_p_Mh.T @ p_dot_d_p_dv

        p_dot_d_p_Xr = p_Mr_p_Xr.T @ p_dot_d_p_Mr
        p_dot_d_p_Xh = p_Mh_p_Xh.T @ p_dot_d_p_Mh

        p_d_p_Mr = np.vstack([ (dp / d)[:,np.newaxis], np.zeros((3,1))])
        p_d_p_Mh = np.vstack([(-dp / d)[:,np.newaxis], np.zeros((3,1))])

        p_d_p_Xr = p_Mr_p_Xr.T @ p_d_p_Mr
        p_d_p_Xh = p_Mh_p_Xh.T @ p_d_p_Mh
        

        p_phi_p_Xr = - 2 * d * p_d_p_Xr - self.k_phi * p_dot_d_p_Xr[:,np.newaxis]
        p_phi_p_Xh = - 2 * d * p_d_p_Xh - self.k_phi * p_dot_d_p_Xh[:,np.newaxis]

        phi = self.dmin**2 + self.eta * self.dynamics.ts + self.lambda0 * self.dynamics.ts - d**2 - self.k_phi * dot_d

        dot_phi = p_phi_p_Xr.T @ dot_Xr + p_phi_p_Xh.T @ dot_Xh

        L = (p_phi_p_Xr.T @ fu)[0]
        S = (-self.eta - self.lambda0 - p_phi_p_Xh.T @ dot_Xh - p_phi_p_Xr.T @ fx)[0]

        if phi < 0 or L @ u_ref <= S:
            return u_ref

        Q_inv = np.eye(2)
        lam = ((L @ u_ref) - S) / (L @ Q_inv @ L.T) # lagrange multiplier, unrelated to lambdaSEA
        c = lam * L @ Q_inv @ L.T
        u = u_ref - (c * ((Q_inv @ L.T) / (L @ Q_inv @ L.T))).reshape([-1,1])

        return u

    def get_u(self, xh, xr_prev, xh_prev):
        # get control that moves robot towards goal
        goal_u = self.dynamics.get_goal_control(self.x, self.goal)

        # project to set of safe controls
        safe_u = self.get_safe_control(xh, xr_prev, xh_prev, goal_u)

        return safe_u

    def step(self, u):
        self.x = self.dynamics.step(self.x, u)
        return self.x

    def copy(self):
        r = Robot(self.x, self.dynamics, self.goal, self.dmin, self.eta, self.k_phi, self.lambda0)
        r.set_goals(self.goals)
        return r