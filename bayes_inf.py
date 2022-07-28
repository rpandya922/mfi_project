import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from robot import Robot
from human import Human
from dynamics import Dynamics, DIDynamics

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
        # n_actions = 16
        # angles = np.linspace(0, 2 * np.pi, num=(n_actions + 1))[:-1]
        # all_actions = []
        # for r in range(1, 20):
        #     actions = np.array([r * np.cos(angles), r * np.sin(angles)]).T
        #     all_actions.append(actions)
        # self.actions = np.vstack(all_actions)
        
        self.actions = np.mgrid[-20:20:41j, -20:20:41j].reshape(2,-1).T

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
        rs = np.array([-np.linalg.norm(state - s) for s in next_states])[:,None]

        # assume optimal trajectory is defined by straight line towards goal, so reward is negative distance from goal
        opt_rewards = np.linalg.norm((next_states - self.thetas[None,:,:]), axis=1)

        # testing
        dists = []
        for s in next_states:
            dd = []
            for i in range(3):
                t = self.thetas[:,[i]]
                dd.append(np.linalg.norm(s - t))
            dists.append(dd)
        dists = np.array(dists)
        assert np.isclose(dists, opt_rewards).all() # passes

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

# TODO: is it better to have Human and BayesHuman inherit from a BaseHuman or just do it this way?
class BayesHuman(Human):
    def __init__(self, x0, dynamics : Dynamics, goals, belief : BayesEstimator, gamma=1):
        super(BayesHuman, self).__init__(x0, dynamics, goals, gamma)
        self.belief = belief

        # goal is initially set randomly
        # g_idx = np.random.randint(self.goals.shape[1])
        # self.goal = self.goals[:,[g_idx]]
        # goal is set to the closest goal to the initial state
        dists = np.linalg.norm(self.goals - x0, axis=0)
        g_idx = np.argmin(dists)
        self.goal = self.goals[:,[g_idx]]

        # TODO: set control limits

    def get_goal(self, get_idx=False):
        r_goal = self.goals[:,[np.argmax(self.belief.belief)]]

        # if the estimated goal of the robot is the same as the human's current goal, change the goal
        if np.linalg.norm(r_goal - self.goal) <= 1e-3:
            # choose a new goal by selecting the closest goal that isn't the same as the robot's
            dists = np.linalg.norm(self.goals - r_goal, axis=0)
            dists[np.argmax(self.belief.belief)] = np.inf
            g_idx = np.argmin(dists)
            
            # g_idx = np.random.randint(self.goals.shape[1])
            self.goal = self.goals[:,[g_idx]]
        
        if get_idx:
            # compute the index of the human's current goal
            g_idx = np.argmin(np.linalg.norm(self.goals - self.goal, axis=0))

            return self.goal, g_idx
        return self.goal

    def update_belief(self, robot_x, robot_u):
        self.belief.update_belief(robot_x, robot_u)

    def copy(self):
        return BayesHuman(self.x, self.dynamics, self.goals, self.belief.copy(), self.gamma)

# TODO: is it better to have Robot and BayesRobot inherit from a BaseRobot or just do it this way?
class BayesRobot(Robot):
    def __init__(self, x0, dynamics : Dynamics, goals, belief : BayesEstimator):
        super(BayesRobot, self).__init__(x0, dynamics, goals)
        self.x = x0
        self.dynamics = dynamics
        self.goals = goals
        self.belief = belief

        # goal is initially set randomly
        g_idx = np.random.randint(self.goals.shape[1])
        self.goal = self.goals[:,[g_idx]]

    def update_belief(self, human_x, human_u):
        self.belief.update_belief(human_x, human_u)

def test_inference():
    # np.random.seed(0)
    goals = np.random.uniform(size=(4, 2))*20 - 10
    goals[[1,3],:] = np.zeros((2, 2))
    print(goals)
    ts = 0.05
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

def full_simulation():
    np.random.seed(1)
    # randomly initialize xh0, xr0, goals
    xh0 = np.random.uniform(size=(4, 1))*20 - 10
    xh0[[1,3]] = np.zeros((2, 1))
    xr0 = np.random.uniform(size=(4, 1))*20 - 10
    xr0[[1,3]] = np.zeros((2, 1))
    goals = np.random.uniform(size=(4, 3))*20 - 10
    goals[[1,3],:] = np.zeros((2, 3))

    ts = 0.05
    h_dynamics = DIDynamics(ts)
    r_dynamics = DIDynamics(ts)

    h_belief = BayesEstimator(goals, r_dynamics, beta=20)
    human = BayesHuman(xh0, h_dynamics, goals, h_belief)

    r_belief = BayesEstimator(goals, h_dynamics, beta=20)
    robot = BayesRobot(xr0, r_dynamics, goals, r_belief)

    fig, ax = plt.subplots()

    # plotting robot belief
    # ax.bar(range(len(r_belief.belief)), r_belief.belief, tick_label=['theta1', 'theta2', 'theta3'])
    # ax.set_ylim(0, 1)
    # ax.set_ylabel("P(theta)")
    # plt.show()
    # 1/0

    # plotting possible actions
    # ax.axis('equal')
    # ax.scatter(r_belief.actions[:,0], r_belief.actions[:,1])

    # ax.set_title("possible actions")
    # ax.set_xlabel("u_1")
    # ax.set_ylabel("u_2")
    # plt.show()
    # 1/0

    for t in range(100):
        # plot human and goals
        ax.cla()
        ax.scatter(human.x[0], human.x[2])
        ax.scatter(robot.x[0], robot.x[2])
        ax.scatter(goals[0], goals[2])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)

        # human & robot decide on actions
        uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # humans and robot observe each others' actions and update beliefs
        human.update_belief(robot.x, ur)
        robot.update_belief(human.x, uh)

        # human and robot move
        human.step(uh)
        robot.step(ur)

    plt.show()

if __name__ == "__main__":
    np.random.seed(1)
    # randomly initialize xh0, xr0, goals
    xh0 = np.random.uniform(size=(4, 1))*20 - 10
    xh0[[1,3]] = np.zeros((2, 1))
    goals = np.random.uniform(size=(4, 3))*20 - 10
    goals[[1,3],:] = np.zeros((2, 3))
    h_goal_idx = 0
    h_goal = goals[:,[h_goal_idx]]

    ts = 0.05
    h_dynamics = DIDynamics(ts)
    human = Human(xh0, h_dynamics, goals)
    r_belief = BayesEstimator(goals, h_dynamics, beta=1)
    print(r_belief.belief)

    fig, ax = plt.subplots()
    # fig2, ax2 = plt.subplots()

    # plotting possible actions
    # ax.axis('equal')
    # ax.scatter(r_belief.actions[:,0], r_belief.actions[:,1])
    # ax.set_title("possible actions")
    # ax.set_xlabel("u_1")
    # ax.set_ylabel("u_2")
    # plt.show()
    # 1/0

    for t in range(100):
        # plot human and goals
        ax.cla()
        ax.scatter(human.x[0], human.x[2], s=200)
        ax.scatter(goals[0], goals[2], c=['#3A637B', '#C4A46B', '#FF5A00'], s=200)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)
        # if t in [0, 33, 67, 99]:
        #     ax2.cla()
        #     ax2.bar(range(len(r_belief.belief)), r_belief.belief, tick_label=['theta1', 'theta2', 'theta3'],
        #         color=['#3A637B', '#C4A46B', '#FF5A00'])
        #     ax2.set_ylim(0, 1)
        #     ax2.set_ylabel("P(theta)")
        #     fig2.savefig(f"./data/belief_{t}.png")
        #     fig.savefig(f"./data/human_{t}.png")

        # human & robot decide on actions
        uh = human.dynamics.get_goal_control(human.x, h_goal)

        # robot observes human's action and update beliefs
        r_belief.update_belief(human.x, uh)
        print(r_belief.belief)

        # human and robot move
        human.step(uh)

    plt.show()
