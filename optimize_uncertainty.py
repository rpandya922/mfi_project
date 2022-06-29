# optimizing for inputs that result in uncertain predictions from intention prediction model

import numpy as np
import torch
import torch.nn as nn
softmax = torch.nn.Softmax(dim=1)
from torch.autograd.functional import jacobian
import cyipopt as ipopt

from dynamics import DIDynamics
from human import Human
from robot import Robot
from intention_predictor import create_model

class MaxEntropyPredictionProblem(object):
    """
    Trajectory Optimization problem where the objective is to find the trajectory of the 
    human and robot that leads to a high-entropy prediction from the intention prediction network
    """
    def __init__(self, human, robot, model, k_hist, k_plan, xh0, xr0):
        self.human = human
        self.robot = robot
        self.model = model
        self.k_hist = k_hist
        self.k_plan = k_plan
        self.xh0 = xh0.flatten() # model uses column vectors, but we use row vectors in optimizer
        self.xr0 = xr0.flatten()
        self.goals = human.goals

        n_human = human.dynamics.n
        m_human = human.dynamics.m
        n_robot = robot.dynamics.n
        m_robot = robot.dynamics.m

        self.n_human = n_human
        self.m_human = m_human
        self.n_robot = n_robot
        self.m_robot = m_robot

        # combine robot history and plan pieces into one block
        k_robot = k_hist + k_plan
        self.k_robot = k_robot
        
        # total number of variables
        n_var = n_human*k_hist + m_human*(k_hist-1) + n_robot*k_robot + m_robot*(k_robot-1)
        
        # indices of different pieces of decision variable
        xh_idx = range(0, n_human*k_hist)
        uh_idx = range(xh_idx[-1]+1, xh_idx[-1]+1+m_human*(k_hist-1))
        xr_idx = range(uh_idx[-1]+1, uh_idx[-1]+1+(n_robot*k_robot))
        ur_idx = range(xr_idx[-1]+1, xr_idx[-1]+1+(m_robot*(k_robot-1)))

        self.n_var = n_var
        self.xh_idx = xh_idx
        self.uh_idx = uh_idx
        self.xr_idx = xr_idx
        self.ur_idx = ur_idx
        
        # TODO: add control constraints
        # total number of constraints
        m_const = n_human + n_robot + n_human*(k_hist-1) + n_robot*(k_robot-1)

        # set up constraint vector indices
        xh0_idx = range(0, n_human)
        xr0_idx = range(xh0_idx[-1]+1, xh0_idx[-1]+1+n_robot)
        xh_dyn_idx = range(xr0_idx[-1]+1, xr0_idx[-1]+1 + (n_human*(k_hist-1)))
        xr_dyn_idx = range(xh_dyn_idx[-1]+1, xh_dyn_idx[-1]+1 + (n_robot*(k_robot-1)))

        self.m_const = m_const
        self.xh0_idx = xh0_idx
        self.xr0_idx = xr0_idx
        self.xh_dyn_idx = xh_dyn_idx
        self.xr_dyn_idx = xr_dyn_idx

    def _get_xh(self, x):
        xh_view = x[self.xh_idx]
        return xh_view.reshape((self.k_hist, self.n_human))

    def _get_uh(self, x):
        uh_view = x[self.uh_idx]
        return uh_view.reshape((self.k_hist-1, self.m_human))

    def _get_xr(self, x):
        xr_view = x[self.xr_idx]
        return xr_view.reshape((self.k_robot, self.n_robot))

    def _get_ur(self, x):
        ur_view = x[self.ur_idx]
        return ur_view.reshape((self.k_robot-1, self.m_robot))

    def objective(self, x):
        xh = self._get_xh(x)
        xr = self._get_xr(x)
        xr_hist = xr[:self.k_hist,:]
        xr_plan = xr[self.k_hist:,:]
        goals = self.goals

        # transform into torch tensors to pass into model
        traj_hist = torch.tensor(np.hstack((xh, xr_hist))).float().unsqueeze(0)
        xr_plan = torch.tensor(xr_plan).float().unsqueeze(0)
        goals = torch.tensor(goals).float().unsqueeze(0)

        probs = softmax(self.model(traj_hist, xr_plan, goals))
        logits = torch.log2(probs)
        entropy = -torch.sum(probs * logits)

        return entropy.item()

    def _torch_objective(self, x):
        # optimizer uses numpy but we need to take the gradient of the objective with torch, code is slightly different
        # TODO: make this a helper function for the normal objective function?
        xh = self._get_xh(x)
        xr = self._get_xr(x)
        xr_hist = xr[:self.k_hist,:]
        xr_plan = xr[self.k_hist:,:]
        goals = self.goals

        # transform into torch tensors to pass into model
        traj_hist = torch.cat((xh, xr_hist), dim=1).unsqueeze(0)
        xr_plan = xr_plan.unsqueeze(0)
        goals = torch.tensor(goals).float().unsqueeze(0)

        probs = softmax(self.model(traj_hist, xr_plan, goals))
        logits = torch.log2(probs)
        entropy = -torch.sum(probs * logits)

        return entropy

    def gradient(self, x):
        x_torch = torch.tensor(x).float()
        grad = jacobian(self._torch_objective, x_torch)

        import ipdb; ipdb.set_trace()
        return grad.detach().numpy()

    def constraints(self, x):
        c_vec = np.zeros(self.m_const)

        # evaluate constraints in order
        xh = self._get_xh(x)
        uh = self._get_uh(x)
        xr = self._get_xr(x)
        ur = self._get_ur(x)

        # xh0
        xh0 = xh[0,:]
        c_vec[self.xh0_idx] = xh0 - self.xh0

        # xr0
        xr0 = xr[0,:]
        c_vec[self.xr0_idx] = xr0 - self.xr0

        # NOTE: it's faster to vectorize like this, but is specific to having LTI dynamics
        # xh dynamics
        xh_next = (self.human.dynamics.A @ xh.T[:,:-1] + self.human.dynamics.B @ uh.T).T
        h_diffs = xh[1:,] - xh_next
        c_vec[self.xh_dyn_idx] = h_diffs.flatten()

        # xr dynamics
        xr_next = (self.robot.dynamics.A @ xr.T[:,:-1] + self.robot.dynamics.B @ ur.T).T
        r_diffs = xr[1:,] - xr_next
        c_vec[self.xr_dyn_idx] = r_diffs.flatten()

        return c_vec

    def jacobian(self, x):
        pass

    def hessian(self, x, lagrange, obj_factor):
        pass

class FC(nn.Module):
    def __init__(self, in_dim=72):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

        self.linear = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
            )

    def forward(self, x1, x2, x3):
        x = torch.cat((x1.flatten(start_dim=1), x2.flatten(start_dim=1), x3.flatten(start_dim=1)), dim=1)
        # return self.fc2(self.relu(self.fc1(x)))
        return self.linear(x)


if __name__ == "__main__":
    model = create_model()
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor.pt"))

    ts = 0.05
    horizon = 100
    k_hist = 5
    k_plan = 5

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

    # model2 = FC(86)
    model2 = FC()

    problem = MaxEntropyPredictionProblem(human, robot, model2, k_hist, k_plan, xh0, xr0)
    x = np.random.uniform(size=problem.n_var)
    # problem.objective(x)
    # problem.constraints(x)
    problem.gradient(x)
