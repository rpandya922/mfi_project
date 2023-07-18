import numpy as np
import torch
from torch.autograd.functional import jacobian, hessian
import cyipopt as ipopt
import pickle

from dynamics import DIDynamics
from intention_predictor import create_model
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

def compute_features_torch(xh_hist, xr_hist, xr_future, goals):
    # computes trajectory features given input torch tensors
    """
    features list (for history traj):
        - distance between human and robot
        - distance of human to each goal
        - distance of robot to each goal
        - angle from human's velocity towards each goal
        - angle from robot's velocity towards each goal
    features list (for robot future traj):
        - distance from robot to each goal
        - angle from robot's velocity towards each goal
    """
    hr_dists = torch.norm(xh_hist[[0,2]] - xr_hist[[0,2]], dim=0, keepdim=True).T
    h_goal_dists = torch.norm(xh_hist.T[:,:,None] - goals, dim=1)
    r_goal_dists = torch.norm(xr_hist.T[:,:,None] - goals, dim=1)
    h_rel = (goals - xh_hist.T[:,:,None])[:,[0,2],:].swapaxes(0,1)
    h_vel = xh_hist[[1,3]]
    r_rel = (goals - xr_hist.T[:,:,None])[:,[0,2],:].swapaxes(0,1)
    r_vel = xr_hist[[1,3]]

    # compute angle between human's velocity and vector to each goal
    h_rel_unit = h_rel / torch.norm(h_rel, dim=0)
    # need to handle zero vectors
    h_vel_norm = torch.norm(h_vel, dim=0)
    h_vel_norm[h_vel_norm == 0] = 1
    h_vel_unit = h_vel / h_vel_norm
    h_angles = torch.vstack([torch.arccos(torch.clip(torch.matmul(h_vel_unit.T, h_rel_unit[:,:,i]).diagonal(), -1.0, 1.0)) for i in range(goals.shape[1])]).T

    # compute angle between robot's velocity and vector to each goal
    r_rel_unit = r_rel / torch.norm(r_rel, dim=0)
    # need to handle zero vectors
    r_vel_norm = torch.norm(r_vel, dim=0)
    r_vel_norm[r_vel_norm == 0] = 1
    r_vel_unit = r_vel / r_vel_norm
    r_angles = torch.vstack([torch.arccos(torch.clip(torch.matmul(r_vel_unit.T, r_rel_unit[:,:,i]).diagonal(), -1.0, 1.0)) for i in range(goals.shape[1])]).T

    r_future_dists = torch.norm(xr_future.T[:,:,None] - goals, dim=1)
    r_future_rel = (goals - xr_future.T[:,:,None])[:,[0,2],:].swapaxes(0,1)
    r_future_vel = xr_future[[1,3]]
    r_future_rel_unit = r_future_rel / torch.norm(r_future_rel, dim=0)
    # need to handle zero vectors
    r_future_vel_norm = torch.norm(r_future_vel, dim=0)
    r_future_vel_norm[r_future_vel_norm == 0] = 1
    r_future_vel_unit = r_future_vel / r_future_vel_norm
    eps = 1e-7 # for numerical stability in gradient computation
    r_future_angles = torch.vstack([torch.arccos(torch.clip(torch.matmul(r_future_vel_unit.T, r_future_rel_unit[:,:,i]).diagonal(), -1.0+eps, 1.0-eps)) for i in range(goals.shape[1])]).T

    # concatenate features
    input_feats = torch.hstack((hr_dists, h_goal_dists, r_goal_dists, h_angles, r_angles))
    future_feats = torch.hstack((r_future_dists, r_future_angles))

    return input_feats.T, future_feats.T

def process_model_input_torch(xh_hist, xr_hist, xr_plan, goals):
    traj_hist = torch.hstack((xh_hist.T, xr_hist.T)).unsqueeze(0)
    xr_plan = xr_plan.unsqueeze(0)
    goals = goals.unsqueeze(0)
    return traj_hist, xr_plan, goals

class TrajOptProb(object):
    def __init__(self, model, stats_file, r_dyn, history=5, horizon=20, hist_feats=21, plan_feats=10):
        self.n = 4
        self.m = 2
        self.model = model
        self.stats_file = stats_file
        self.history = history
        self.horizon = horizon
        self.hist_feats = hist_feats
        self.plan_feats = plan_feats
        self.r_dyn = r_dyn
        self.A = torch.tensor(r_dyn.A).float()
        self.B = torch.tensor(r_dyn.B).float()

        self.n_state = self.horizon*self.n
        self.n_ctrl = self.horizon*self.m

        self.stats = pickle.load(open(self.stats_file, "rb"))
        # convert stats to torch tensors
        for key, val in self.stats.items():
            self.stats[key] = torch.tensor(val).float()

        self.goals = None
        self.xh_hist = None
        self.xr_hist = None

    def _set_nn_inputs(self, goals, xh_hist, xr_hist):
        # so that we can re-use the same problem instance for different timesteps
        self.goals = torch.tensor(goals).float()
        self.xh_hist = torch.tensor(xh_hist).float()
        self.xr_hist = torch.tensor(xr_hist).float()
        self.xr_prev = torch.tensor(xr_hist[:,[-1]]).float() # used in constraint computation

    def _process_input(self, x):
        # x should already be torch tensor (to make gradient computation easy)
        # first, split into states and controls
        xr_plan = x[:self.n_state].reshape((self.n, self.horizon))

        # compute NN inputs
        hist_feats, future_feats = compute_features_torch(self.xh_hist, self.xr_hist, xr_plan, self.goals)
        input_hist, input_future, input_goals = process_model_input_torch(self.xh_hist, self.xr_hist, xr_plan.T, self.goals)
        input_hist = torch.cat((input_hist, hist_feats.T.unsqueeze(0)), dim=2)
        input_future = torch.cat((input_future, future_feats.T.unsqueeze(0)), dim=2)

        input_hist = (input_hist - self.stats["input_traj_mean"]) / self.stats["input_traj_std"]
        input_future = (input_future - self.stats["robot_future_mean"]) / self.stats["robot_future_std"]
        input_goals = (input_goals.transpose(1,2) - self.stats["input_goals_mean"]) / self.stats["input_goals_std"]
        input_goals = input_goals.transpose(1,2)

        return input_hist, input_future, input_goals

    def _obj_helper(self, input_hist, input_future, input_goals):
        # query model
        model_out = self.model(input_hist, input_future, input_goals)
        probs = softmax(model_out)
        
        P = probs.flatten()
        # Q = torch.ones(3) / 3
        # return (P * (P / Q).log()).sum() # kl divergence
        return -P[1] # probability of first goal
        # return 0*P[0]

    def objective(self, x):
        x = torch.tensor(x).float()
        inputs = self._process_input(x)
        return self._obj_helper(*inputs).detach().numpy().item()

    def gradient(self, x):
        x_ = torch.tensor(x).float()
        x_.requires_grad = True
        input_hist, input_future, input_goals = self._process_input(x_)
        obj_val = self._obj_helper(input_hist, input_future, input_goals)
        obj_val.backward()

        return x_.grad.detach().numpy()
    
    def _const_helper(self, x):
        xr_plan = x[:self.n_state].reshape((self.n, self.horizon))
        ur_plan = x[self.n_state:].reshape((self.m, self.horizon))
        xr_plan_prev = torch.hstack((self.xr_prev, xr_plan[:,:-1]))
        dyn_step = self.A @ xr_plan_prev + self.B @ ur_plan
        dyn_const = torch.norm(xr_plan - dyn_step, dim=0)

        return dyn_const

    def constraints(self, x):
        x_ = torch.tensor(x).float()
        const = self._const_helper(x_)
        return const.detach().numpy()
    
    # for old jacobian (constraints included control constraint, now moved to bounds on x)
    # def jacobianstructure(self):
    #     rows = np.array([ 0,  0,  1,  1,  1,  1,  3,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,
    #                       7,  7,  7,  7,  9,  9,  9,  9, 10, 10, 10, 10, 12, 12, 12, 12, 13, 13,
    #                       13, 13, 16, 16, 16, 16, 16, 16, 16, 16, 19, 19, 19, 19, 20, 21, 22, 23,
    #                       24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    #                       42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
    #     cols = np.array([ 0,  80,  40,  41,  60, 101,   2,   3,  22,  83,   3,   4,  23,  84,
    #                       44,  45,  64, 105,   6,   7,  26,  87,  48,  49,  68, 109,   9,  10,
    #                       29,  90,  51,  52,  71, 112,  12,  13,  32,  93,  15,  16,  35,  55,
    #                       56,  75,  96, 116,  18,  19,  38,  99,  80,  81,  82,  83,  84,  85,
    #                       86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
    #                       100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    #                       114, 115, 116, 117, 118, 119])
    #     return (rows, cols)

    def jacobian(self, x):
        # the constraint jacobian
        # TODO: manually construct sparse jacobian
        x_ = torch.tensor(x).float()
        const_fn = lambda x: self._const_helper(x)
        jac = jacobian(const_fn, x_)

        # return jac.detach().numpy()[self.jacobianstructure()]
        return jac.detach().numpy().flatten()

    # TODO: at least remove hessian dependence on ur_plan (it's always zero)
    # def hessianstructure(self):
    #     pass

    # NOTE: much faster to use IPOPT's AD to compute hessian
    # def hessian(self, x, lagrange, obj_factor):
    #     # objective hessian
    #     x_ = torch.tensor(x).float()
    #     obj_fn = lambda x: self._obj_helper(*self._process_input(x))
    #     obj_hess = hessian(obj_fn, x_).detach().numpy()

    #     # constraint hessian
    #     # TODO: manually construct sparse constraint hessian
    #     const_fn = lambda x: self._const_helper(x)
    #     const_jac = lambda x: jacobian(const_fn, x)
    #     const_hess = jacobian(const_jac, x_).detach().numpy()

    #     hess = obj_factor*obj_hess + np.tensordot(lagrange, const_hess, axes=([0],[0]))
    #     return hess.flatten()

if __name__ == "__main__":
    horizon = 20
    hidden_size = 128
    num_layers = 2
    hist_feats = 21
    plan_feats = 10
    k_hist=5
    k_plan=20
    model_path = "./data/models/prob_pred_intention_predictor_bayes_20230630-164246.pt"
    predictor = create_model(horizon_len=horizon, hidden_size=hidden_size, num_layers=num_layers, hist_feats=hist_feats, plan_feats=plan_feats)
    predictor.load_state_dict(torch.load(model_path, map_location=device))
    # TODO: fetch this file from remote server for local testing
    stats_file = "./data/prob_pred/bayes_prob_branching_processed_feats_stats.pkl"

    # generate initial conditions
    np.random.seed(2)
    ts = 0.05
    xh0 = np.random.uniform(-10, 10, (4, 1))
    xh0[[1,3]] = 0
    xr0 = np.random.uniform(-10, 10, (4, 1))
    xr0[[1,3]] = 0
    goals = np.random.uniform(-10, 10, (4, 3))
    goals[[1,3]] = 0
    r_goal = goals[:,[0]]

    # create human and robot objects
    W = np.diag([0.0, 0.7, 0.0, 0.7])
    # W = np.diag([0.0, 0.0, 0.0, 0.0])
    h_dynamics = DIDynamics(ts=ts, W=W)
    r_dynamics = DIDynamics(ts=ts)

    # create robot and human objects
    h_belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.7)
    human = BayesHuman(xh0, h_dynamics, goals, h_belief, gamma=5)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)

    # create trajopt problem 
    obj = TrajOptProb(predictor, stats_file, r_dynamics, history=k_hist, horizon=k_plan, hist_feats=hist_feats, plan_feats=plan_feats)

    # data saving
    xh_traj = xh0
    xr_traj = xr0

    # simulate system forward for k_hist timesteps
    for k in range(k_hist):
        # compute agent controls
        uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # update human's belief
        human.update_belief(robot.x, ur)

        # step forward
        xh = human.step(uh)
        xr = robot.step(ur)

        # save data
        xh_traj = np.hstack((xh_traj, xh))
        xr_traj = np.hstack((xr_traj, xr))
    
    obj._set_nn_inputs(goals, xh_traj[:,-k_hist:], xr_traj[:,-k_hist:])

    # TODO: move into object
    n_state = k_plan*4
    n_ctrl = k_plan*2
    # construct xr_plan by inputting 0 control for k_plan timesteps
    xr_plan = np.zeros((4, k_plan))
    ur_plan = np.zeros((2, k_plan))
    xr_sim = robot.x
    for i in range(k_plan):
        # u = np.zeros((2,1))
        u = robot.dynamics.get_goal_control(xr_sim, robot.goal)
        xr_sim = robot.dynamics.step(xr_sim, u)
        xr_plan[:,i] = xr_sim.flatten()
        ur_plan[:,i] = u.flatten()

    # construct initial guess
    x0 = np.hstack((xr_plan.flatten(), ur_plan.flatten()))

    val = obj.objective(x0)
    grad = obj.gradient(x0)
    const = obj.constraints(x0)
    jac = obj.jacobian(x0)
    # hess = obj.hessian(x0, np.ones(const.shape[0]), 1.0)
    cl = np.zeros((k_plan,))
    cu = np.zeros((k_plan,))
    lb = np.hstack((np.tile(-np.inf, n_state), -10*np.ones(n_ctrl)))
    ub = np.hstack((np.tile(np.inf, n_state), 10*np.ones(n_ctrl)))
    nlp = ipopt.Problem(
            n=len(x0),
            m=len(cl),
            problem_obj=obj,
            cl=cl,
            cu=cu,
            lb=lb,
            ub=ub
            )
    # nlp.addOption('mu_strategy', 'adaptive')
    # nlp.addOption('tol', 1e-5)
    # nlp.addOption('constr_viol_tol', 1e-4)
    nlp.addOption('max_iter', 1000)
    # NOTE: runs but does not converge
    # converges with objective fn = 0, but takes ~180 iterations even though initial guess is feasible (?)
    x, info = nlp.solve(x0)
    import ipdb; ipdb.set_trace()
