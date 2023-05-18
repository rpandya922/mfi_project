import numpy as np
from scipy.special import softmax

class BayesEstimator():
    def __init__(self, thetas, dynamics, prior=None, beta=0.7):

        self.thetas = thetas
        self.dynamics = dynamics
        self.beta = beta

        n_theta = thetas.shape[1]
        if prior is None:
            prior = np.ones(n_theta) / n_theta
        else:
            assert len(prior) == n_theta
        self.belief = prior

        # define possible actions (for the purpose of inference, we discretize the actual action taken by the agent)
        n_actions = 8
        angles = np.linspace(0, 2 * np.pi, num=(n_actions + 1))[:-1]
        all_actions = []
        for r in range(1, 2):
            actions = np.array([r * np.cos(angles), r * np.sin(angles)]).T
            all_actions.append(actions)
        self.actions = np.vstack(all_actions)
        
        # self.actions = np.mgrid[-20:20:41j, -20:20:41j].reshape(2,-1).T

    def project_action(self, action):
        # passed-in action will be a column vector
        a = action.flatten()
        # find closest action
        dists = np.linalg.norm(self.actions - a, axis=1)
        a_idx = np.argmin(dists)

        return self.actions[a_idx], a_idx

    def update_belief(self, state, action):
        # project chosen action to discrete set
        _, a_idx = self.project_action(action)

        # consider the next state if each potential action was chosen
        next_states = np.array([self.dynamics.step_mean(state, a[:,None]) for a in self.actions]) # dynamics.step expects column vectors
        rs = np.array([-np.linalg.norm(state - s) for s in next_states])[:,None]

        # assume optimal trajectory is defined by straight line towards goal, so reward is negative distance from goal
        opt_rewards = np.linalg.norm((next_states - self.thetas[None,:,:]), axis=1)

        # testing
        # dists = []
        # for s in next_states:
        #     dd = []
        #     for i in range(3):
        #         t = self.thetas[:,[i]]
        #         dd.append(np.linalg.norm(s - t))
        #     dists.append(dd)
        # dists = np.array(dists)
        # assert np.isclose(dists, opt_rewards).all() # passes

        Q_vals = rs - opt_rewards

        # compute probability of choosing each action
        prob_action = softmax(self.beta * Q_vals, axis=0)
        # get row corresponding to chosen action
        y_i = prob_action[a_idx]

        # update belief
        new_belief = (y_i * self.belief) / np.sum(y_i * self.belief)
        self.belief = new_belief

        return new_belief

    def copy(self):
        return BayesEstimator(self.thetas.copy(), self.dynamics, self.belief.copy(), self.beta)