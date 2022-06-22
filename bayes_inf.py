import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from dynamics import DIDynamics

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
        n_actions = 16
        angles = np.linspace(0, 2 * np.pi, num=(n_actions + 1))[:-1]
        all_actions = []
        for r in range(1, 20):
            actions = np.array([r * np.cos(angles), r * np.sin(angles)]).T
            all_actions.append(actions)

        self.actions = np.vstack(all_actions)

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
        next_states = np.array([self.dynamics.step(state, a[:,None]) for a in self.actions]) # dynamics.step expects column vectors
        rs = np.array([-np.linalg.norm(state[[0,2]] - s[[0,2]]) for s in next_states])[:,None]

        # assume optimal trajectory is defined by straight line towards goal, so reward is negative distance from goal
        opt_rewards = np.linalg.norm((next_states - self.thetas[None,:,:])[:,[0,2],:], axis=1)
        Q_vals = rs - opt_rewards

        # compute probability of choosing each action
        prob_action = softmax(self.beta * Q_vals, axis=0)
        # get row corresponding to chosen action
        y_i = prob_action[a_idx]

        # update belief
        new_belief = (y_i * self.belief) / np.sum(y_i * self.belief)
        self.belief = new_belief

        return new_belief

if __name__ == "__main__":
    # np.random.seed(0)
    goals = np.random.uniform(size=(4, 2))*20 - 10
    goals[[1,3],:] = np.zeros((2, 2))
    print(goals)
    ts = 0.05
    # TODO: make Bayesian human object that's doing inference 
    dynamics = DIDynamics(ts)
    xh = np.zeros((4,1))
    g_idx = 1

    b = BayesEstimator(goals, dynamics, beta=5)

    fig, ax = plt.subplots()

    for t in range(100):
        # plot human and goals
        ax.cla()
        ax.scatter(xh[0], xh[2])
        ax.scatter(goals[0], goals[2])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)

        # human moves
        uh = dynamics.get_goal_control(xh, goals[:,[g_idx]])
        xh_new = dynamics.step(xh, uh)

        # update belief based on observed action
        b.update_belief(xh, uh)

        # store new state
        xh = xh_new

        print(b.belief)
        # input(": ")
