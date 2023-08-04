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

    def update_belief_continuous(self, state, action):
        # for each theta, compute probability of choosing each action
        likelihoods = np.zeros_like(self.belief)
        for theta_idx in range(self.thetas.shape[1]):
            theta = self.thetas[:,[theta_idx]]
            Q_val = -(state - theta).T @ (self.dynamics.Q + self.dynamics.P) @ (state - theta) - (action.T @ self.dynamics.R @ action)

            H = 2*self.dynamics.R
            factor = np.sqrt((2*np.pi)**self.dynamics.m / np.linalg.det(H))
            Q_star = -(state - theta).T @ (self.dynamics.Q + self.dynamics.P) @ (state - theta)

            likelihoods[theta_idx] = 1/factor * np.exp(self.beta*(Q_val-Q_star))
        
        # update belief using likelihood
        new_belief = (likelihoods * self.belief) / np.sum(likelihoods * self.belief)
        # import ipdb; ipdb.set_trace()
        # make sure no values are < 0.01
        # new_belief[new_belief < 0.01] = 0.01
        # new_belief = new_belief / np.sum(new_belief)

        self.belief = new_belief

        return new_belief

    def update_belief(self, state, action):
        """
        new method that computes exact likelihoods using LQR cost-to-go and Gaussian integral per goal
        """
        # for each theta, compute probability of choosing each action
        likelihoods = np.zeros_like(self.belief)
        for theta_idx in range(self.thetas.shape[1]):
            theta = self.thetas[:,[theta_idx]]
            next_state = self.dynamics.step_mean(state, action)
            Q_val = -(state - theta).T @ self.dynamics.Q @ (state - theta) - (action.T @ self.dynamics.R @ action) - (next_state - theta).T @ self.dynamics.Pd @ (next_state - theta)

            H = 2*self.dynamics.R + self.dynamics.B.T@self.dynamics.Pd@self.dynamics.B
            factor = np.sqrt((2*np.pi)**self.dynamics.m / np.linalg.det(H))
            Q_star = -(state - theta).T @ self.dynamics.Pd @ (state - theta)

            likelihoods[theta_idx] = 1/factor * np.exp(self.beta*(Q_val-Q_star))

        # update belief using likelihood
        new_belief = (likelihoods * self.belief) / np.sum(likelihoods * self.belief)
        
        # make sure no values are < 0.01
        new_belief[new_belief < 0.01] = 0.01
        new_belief = new_belief / np.sum(new_belief)

        self.belief = new_belief

        return new_belief

    def copy(self):
        return BayesEstimator(self.thetas.copy(), self.dynamics, self.belief.copy(), self.beta)