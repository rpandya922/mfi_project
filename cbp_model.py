# model-based conditional behavior prediction
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

class CBPEstimator():
    def __init__(self, thetas, dynamics, prior=None, beta=0.7):
        self.thetas = thetas # column vectors
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
        all_actions.append(np.array([0, 0]).T)
        self.actions = np.vstack(all_actions)

        # pre-computing scores for every 3 goal combinations
        # scores = np.zeros((self.thetas.shape[1], self.thetas.shape[1], self.thetas.shape[1]))
        # for i in range(self.thetas.shape[1]):
        #     for j in range(self.thetas.shape[1]):
        #         for k in range(self.thetas.shape[1]):
        #             scores[i,j,k] = self.score_thetas(self.thetas[:,[i]], self.thetas[:,[j]], self.thetas[:,[k]])
        # self.scores = scores

    
    def project_action(self, action):
        # passed-in action will be a column vector
        a = action.flatten()
        # find closest action
        dists = np.linalg.norm(self.actions - a, axis=1)
        a_idx = np.argmin(dists)

        return self.actions[a_idx], a_idx

    def score_thetas(self, theta1, theta2, theta_prior, state, w=2):
        """
        generally, theta1 is the considered new goal for the human, theta2 is the robot's chosen goal, and theta_prior is the human's current goal (or our consideration of it)
        """
        # we want to model the behavior that the human chooses the goal closest to them that's not the same as the robot's goal
        # high score -> high interaction, so we want a low score for theta1 being close to state 
        s = 2*np.linalg.norm(theta1 - state) - np.linalg.norm(theta1 - theta2) + w*np.linalg.norm(theta1 - theta_prior)
        # NOTE: converges to original belief as w -> inf (this means we think human won't be very affected by robot's choice)
        # s = -np.linalg.norm(theta1 - theta2) + w*np.linalg.norm(theta1 - theta_prior)
        return s
    
    def weight_by_score(self, prior_belief, theta_r, state, beta=1):
        theta_r_idx = np.argmin(np.linalg.norm(self.thetas - theta_r, axis=0))
        belief_post = np.zeros(self.thetas.shape[1])
        for i in range(self.thetas.shape[1]):
            # don't directly need theta_post value, just index (values computed for all thetas)
            # theta_post = self.thetas[:,[i]]
            likelihood = np.zeros(self.thetas.shape[1])
            for j in range(self.thetas.shape[1]):
                theta_prior = self.thetas[:,[j]] # don't need this, just index
                scores = [self.score_thetas(self.thetas[:,[k]], theta_r, theta_prior, state) for k in range(self.thetas.shape[1])]
                # scores = self.scores[:,theta_r_idx,j]
                # assert np.allclose(scores, scores2) # passes
                p_post_given_prior = softmax(-beta * np.array(scores))[i]
                likelihood[j] = p_post_given_prior # * prior_belief[j]
            belief_post[i] = likelihood @ prior_belief # np.sum(likelihood)

        # vectorized version
        # p_post = softmax(-beta * self.scores[:,:,theta_r_idx], axis=0)
        # belief_post2 = np.sum(p_post * prior_belief[:,None], axis=0)
        # import ipdb; ipdb.set_trace()
        # assert np.allclose(belief_post, belief_post2)

        return belief_post

    def update_belief(self, state, action, return_likelihood=False):
        """
        new method that computes exact likelihoods using LQR cost-to-go and Gaussian integral per goal
        """
        # for each theta, compute probability of choosing each action
        likelihoods = np.zeros_like(self.belief)
        for theta_idx in range(self.thetas.shape[1]):
            theta = self.thetas[:,[theta_idx]]
            next_state = self.dynamics.step(state, action)
            Q_val = -(state - theta).T @ self.dynamics.Q @ (state - theta) - (action.T @ self.dynamics.R @ action) - (next_state - theta).T @ self.dynamics.P @ (next_state - theta)

            H = 2*self.dynamics.R + self.dynamics.B.T@self.dynamics.P@self.dynamics.B
            factor = np.sqrt((2*np.pi)**self.dynamics.m / np.linalg.det(H))
            Q_star = -(state - theta).T @ self.dynamics.P @ (state - theta)

            likelihoods[theta_idx] = 1/factor * np.exp(self.beta*(Q_val-Q_star))

        # update belief using likelihood
        new_belief = (likelihoods * self.belief) / np.sum(likelihoods * self.belief)
        self.belief = new_belief

        if not return_likelihood:
            return new_belief
        else:
            return new_belief, likelihoods

    def update_belief_old(self, state, action, r_state, return_likelihood=False):
        # project chosen action to discrete set
        _, a_idx = self.project_action(action)

        # robot should only have access to mean/non-noisy dynamics and also include term in control for robot's state
        # consider the next state if each potential action was chosen
        step = lambda u: self.dynamics.A @ state + self.dynamics.B @ (u + self.dynamics.gamma/np.linalg.norm(state - r_state)**2*self.dynamics.get_robot_control(state, r_state))
        next_states = np.array([step(a[:,None]) for a in self.actions]) # dynamics.step expects column vectors
        rs = np.array([-np.linalg.norm(state - s) for s in next_states])[:,None]

        # assume optimal trajectory is defined by straight line towards goal, so reward is negative distance from goal
        opt_rewards = np.linalg.norm((next_states - self.thetas[None,:,:]), axis=1)

        Q_vals = rs - opt_rewards

        # compute probability of choosing each action
        prob_action = softmax(self.beta * Q_vals, axis=0)
        # get row corresponding to chosen action
        y_i = prob_action[a_idx]

        # update belief
        new_belief = (y_i * self.belief) / np.sum(y_i * self.belief)
        # self.belief = new_belief

        if not return_likelihood:
            return new_belief
        else:
            return new_belief, y_i

    def copy(self):
        return CBPEstimator(self.thetas.copy(), self.dynamics, self.belief.copy(), self.beta)

class BetaBayesEstimator():
    def __init__(self, thetas, betas, dynamics, prior=None):
        self.thetas = thetas # column vectors
        self.betas = np.array(betas)
        self.dynamics = dynamics

        n_theta = thetas.shape[1]
        n_beta = len(betas)
        if prior is None:
            prior = np.ones((n_theta, n_beta)) / (n_theta*n_beta)
        else:
            assert prior.shape == (n_theta,n_beta)
        self.belief = prior

        # define possible actions (for the purpose of inference, we discretize the actual action taken by the agent)
        n_actions = 8
        angles = np.linspace(0, 2 * np.pi, num=(n_actions + 1))[:-1]
        all_actions = []
        for r in range(1, 2):
            actions = np.array([r * np.cos(angles), r * np.sin(angles)]).T
            all_actions.append(actions)
        # all_actions.append(np.array([0, 0]).T)
        self.actions = np.vstack(all_actions)

        # self.actions = np.mgrid[-20:20:41j, -20:20:41j].reshape(2,-1).T

    def project_action(self, action):
        # passed-in action will be a column vector
        a = action.flatten()
        # find closest action
        dists = np.linalg.norm(self.actions - a, axis=1)
        a_idx = np.argmin(dists)

        return self.actions[a_idx], a_idx
    
    def update_belief_(self, state, action, r_state):
        """
        new method that computes exact likelihoods using LQR cost-to-go and Gaussian integral per goal
        """
        # for each theta, compute probability of choosing each action
        likelihoods = np.zeros_like(self.belief)
        for theta_idx in range(self.thetas.shape[1]):
            theta = self.thetas[:,[theta_idx]]
            for beta_idx in range(self.betas.shape[0]):
                beta = self.betas[beta_idx]
                next_state = self.dynamics.step(state, action)
                Q_val = -(state - theta).T @ self.dynamics.Q @ (state - theta) - (action.T @ self.dynamics.R @ action) - (next_state - theta).T @ self.dynamics.P @ (next_state - theta)

                H = 2*self.dynamics.R + self.dynamics.B.T@self.dynamics.P@self.dynamics.B
                factor = np.sqrt((2*np.pi)**self.dynamics.m / np.linalg.det(H))
                Q_star = -(state - theta).T @ self.dynamics.P @ (state - theta)
                likelihoods[theta_idx, beta_idx] = 1/factor * np.exp(beta*(Q_val-Q_star))

        # update belief using likelihood
        new_belief = (likelihoods * self.belief) / np.sum(likelihoods * self.belief)
        # self.belief = new_belief

        return new_belief

    def update_belief_old_vector(self, state, action, r_state):
        # project chosen action to discrete set
        _, a_idx = self.project_action(action)

        # robot should only have access to mean/non-noisy dynamics and also include term in control for robot's state
        # consider the next state if each potential action was chosen
        step = lambda u: self.dynamics.A @ state + self.dynamics.B @ (u + self.dynamics.gamma/np.linalg.norm(state - r_state)**2*self.dynamics.get_robot_control(state, r_state))
        next_states = np.array([step(a[:,None]) for a in self.actions]) # dynamics.step expects column vectors
        rs = np.array([-np.linalg.norm(state - s) for s in next_states])[:,None]

        # assume optimal trajectory is defined by straight line towards goal, so reward is negative distance from goal
        opt_rewards = np.linalg.norm((next_states - self.thetas[None,:,:]), axis=1)

        Q_vals = rs - opt_rewards
        Q_vals = np.tile(Q_vals[:,:,None], (1,1,self.betas.shape[0])) # mofiying so we can do computation with all betas

        # compute probability of choosing each action (flatten for softmax)
        prob_action = softmax((Q_vals * self.betas).reshape(self.actions.shape[0], self.belief.shape[0]*self.belief.shape[1]), axis=0)
        # get row corresponding to chosen action (unflatten for belief update)
        y_i = prob_action[a_idx]
        y_i = y_i.reshape(self.belief.shape[0],self.belief.shape[1])

        # update belief
        new_belief = (y_i * self.belief) / np.sum(y_i * self.belief)

        # self.belief = new_belief

        return new_belief
    
    def update_belief_old(self, state, action, r_state):
        # project chosen action to discrete set
        _, a_idx = self.project_action(action)
        
        # testing full bayesian belief update "manually" i.e. without vectorizing, for debugging
        new_belief = np.zeros(self.belief.shape)
        for theta_idx in range(self.thetas.shape[1]):
            for beta_idx in range(self.betas.shape[0]):
                theta = self.thetas[:,[theta_idx]]
                beta = self.betas[beta_idx]
                # robot should only have access to mean/non-noisy dynamics and also include term in control for robot's state
                # consider the next state if each potential action was chosen
                step = lambda u: self.dynamics.A @ state + self.dynamics.B @ (u + self.dynamics.gamma/np.linalg.norm(state - r_state)**2*self.dynamics.get_robot_control(state, r_state))
                next_states = np.array([step(a[:,None]) for a in self.actions])
                
                rewards = -np.linalg.norm((next_states - theta), axis=1)
                p_a_given_theta_beta = softmax(beta * rewards)[a_idx][0]
                new_belief[theta_idx, beta_idx] = p_a_given_theta_beta * self.belief[theta_idx, beta_idx]
        
        return new_belief / np.sum(new_belief)


def test_fixed_goal(xh0, xr0, goals, r_goal, h_dynamics, r_dynamics):
    h_belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.7)
    human = BayesHuman(xh0, h_dynamics, goals, h_belief, gamma=5)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    r_belief = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=1)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=1)

    # simulate trajectory
    xh_traj = xh0
    xr_traj = xr0
    # simulate for T seconds
    N = int(15 / ts)
    h_beliefs = h_belief.belief
    r_beliefs = r_belief.belief
    r_beliefs_nominal = r_belief_nominal.belief
    h_goal_idxs = []
    r_goal_idxs = []

    for idx in range(N):
        # compute agent controls
        uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # update human's belief
        human.update_belief(robot.x, ur)
        # simulate robot nominal belief update
        r_belief_nominal.belief = r_belief_nominal.update_belief(human.x, uh, robot.x)
        # update robot's belief with cbp
        r_belief_prior = r_belief.update_belief(human.x, uh, robot.x)
        
        r_belief.belief = r_belief_prior

        # state = human.dynamics.A @ human.x + human.dynamics.B @ uh
        # r_belief_post = r_belief.weight_by_score(r_belief_prior, r_goal, state, beta=0.1)
        # r_belief.belief = r_belief_post

        # step dynamics forward
        xh0 = human.step(uh)
        xr0 = robot.step(ur)

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))
        h_beliefs = np.vstack((h_beliefs, h_belief.belief))
        r_beliefs = np.vstack((r_beliefs, r_belief.belief))
        r_beliefs_nominal = np.vstack((r_beliefs_nominal, r_belief_nominal.belief))
        # save human's actual intended goal
        h_goal_idxs.append(np.argmin(np.linalg.norm(human.goal - goals, axis=0)))
        # save robot's actual intended goal
        r_goal_idxs.append(np.argmin(np.linalg.norm(robot.goal - goals, axis=0)))
    
    # get human's actual final goal
    h_goal_idx = h_goal_idxs[-1]
    # check the probability robot assigns to this goal over time
    h_goal_belief = r_beliefs[:,h_goal_idx]
    # check the first timestep where this goal is assigned > 0.5 probability
    h_goal_timestep = np.argmax(h_goal_belief > 0.5)
    if h_goal_timestep == 0:
        # human's goal was never assigned > 0.5 probability
        h_goal_timestep = np.inf
    # compute the average probability assigned to this goal
    h_goal_prob = np.mean(h_goal_belief)
    # compute if this is the most likely goal at the end
    h_goal_most_likely = np.argmax(r_beliefs[-1]) == h_goal_idx

    return h_goal_idx, h_goal_timestep, h_goal_prob, h_goal_most_likely

def test_seeds():
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from dynamics import DIDynamics
    from bayes_inf import BayesEstimator, BayesHuman  
    from robot import Robot
    from intention_utils import overlay_timesteps

    # generate initial conditions
    all_stats = []
    failed_seeds = []
    for seed in tqdm(range(100)):
        seed_stats = []
        for r_goal_idx in range(3):
            np.random.seed(seed)
            ts = 0.1
            xh0 = np.random.uniform(-10, 10, (4, 1))
            xh0[[1,3]] = 0
            xr0 = np.random.uniform(-10, 10, (4, 1))
            xr0[[1,3]] = 0
            goals = np.random.uniform(-10, 10, (4, 3))
            goals[[1,3]] = 0
            r_goal = goals[:,[r_goal_idx]]

            # create human and robot objects
            W = np.diag([0.0, 0.7, 0.0, 0.7])
            # W = np.diag([0.0, 0.0, 0.0, 0.0])
            h_dynamics = DIDynamics(ts=ts, W=W)
            r_dynamics = DIDynamics(ts=ts)

            idx, timestep, prob, most_likely = test_fixed_goal(xh0, xr0, goals, r_goal, h_dynamics, r_dynamics)
            if not most_likely:
                failed_seeds.append(seed)
                # continue
            if timestep == np.inf:
                timestep = 150
            seed_stats.append((idx, timestep, prob, most_likely))
        all_stats.append(seed_stats)
    flat_stats = np.array([s for seed_stats in all_stats for s in seed_stats])
    print(f"timestep: {flat_stats[:,1].mean()}, prob: {flat_stats[:,2].mean()}, most likely: {flat_stats[:,3].mean()}")

def plot_rollout():
    import matplotlib.pyplot as plt
    from dynamics import DIDynamics
    from bayes_inf import BayesEstimator, BayesHuman  
    from robot import Robot
    from intention_utils import overlay_timesteps

    # generate initial conditions
    np.random.seed(0)
    ts = 0.1
    xh0 = np.random.uniform(-10, 10, (4, 1))
    xh0[[1,3]] = 0
    xr0 = np.random.uniform(-10, 10, (4, 1))
    xr0[[1,3]] = 0
    goals = np.random.uniform(-10, 10, (4, 3))
    goals[[1,3]] = 0
    r_goal = goals[:,[2]] # this is arbitrary since it'll be changed in simulations later anyways

    # create human and robot objects
    # W = np.diag([0.0, 0.7, 0.0, 0.7])
    W = np.diag([0.0, 0.0, 0.0, 0.0])
    h_dynamics = DIDynamics(ts=ts, W=W)
    r_dynamics = DIDynamics(ts=ts)

    h_belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=0.0005)
    human = BayesHuman(xh0, h_dynamics, goals, h_belief, gamma=5)
    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    r_belief = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=0.0005)
    # r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[0.01, 0.1, 1, 10, 100, 1000], dynamics=h_dynamics)
    r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1], dynamics=h_dynamics)

    # simulate trajectory
    xh_traj = xh0
    xr_traj = xr0
    # simulate for T seconds
    N = int(10 / ts)
    h_beliefs = h_belief.belief
    r_beliefs = r_belief.belief
    r_beliefs_nominal = r_belief_nominal.belief
    r_beliefs_beta = r_belief_beta.belief
    r_belief_likelihoods = []
    h_goal_idxs = []
    r_goal_idxs = []

    # figure for plotting
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    axes = np.array(axes).flatten()
    ax = axes[0]
    # make ax equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    h_belief_ax = axes[1]
    r_belief_ax = axes[2]
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
    h_goal_ax = axes[3]
    r_beta_ax = axes[4]
    r_likelihoods_ax = axes[5]

    for idx in range(N):
        # human.goal = np.array([[-10, 0, 10, 0]]).T
        # compute agent controls
        # uh = human.get_u(robot.x)
        uh = human.dynamics.get_goal_control(human.x, human.get_goal())
        # uh = np.zeros((2,1)) #+ np.random.uniform(-5, 5, (2,1))
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # update human's belief
        human.update_belief(robot.x, ur)
        # simulate robot nominal belief update
        # r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, robot.x, return_likelihood=True)
        r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, return_likelihood=True)
        # update robot's belief with cbp
        # r_belief_prior = r_belief.update_belief(human.x, uh, robot.x)
        r_belief_prior = r_belief_nominal.belief
        # r_belief_prior = np.random.uniform(0, 1, (3,)) # TODO: find out why this doesn't change the posterior belief much
        # r_belief_prior = r_belief_prior / np.sum(r_belief_prior)
        # state = human.dynamics.A @ human.x + human.dynamics.B @ uh
        # r_belief_post = r_belief.weight_by_score(r_belief_prior, r_goal, state, beta=0.1)
        # r_belief.belief = r_belief_post
        
        # update the robot's belief over goals and betas
        # r_belief_beta1 = r_belief_beta.update_belief(human.x, uh, robot.x)
        r_belief_beta2 = r_belief_beta.update_belief_(human.x, uh, robot.x)
        # print(np.allclose(r_belief_beta1, r_belief_beta2))
        # print(np.sum(r_belief_beta1 - r_belief_beta2))
        r_belief_beta.belief = r_belief_beta2
        # max_theta_idx = np.argmax(r_belief_beta.belief.sum(axis=1))
        # print(r_belief_beta.belief[max_theta_idx])
        # print(r_belief_beta.betas[np.argmax(r_belief_beta.belief[max_theta_idx])])

        if idx > -1:
            # input(": ")
            # simulate human's next state
            state = human.dynamics.A @ human.x + human.dynamics.B @ uh
            # loop through goals and compute belief update for each
            divs = []
            posts = []
            for goal_idx in range(goals.shape[1]):
                goal = goals[:,[goal_idx]]
                # compute CBP belief update
                r_belief_post = r_belief.weight_by_score(r_belief_prior, goal, state, beta=0.5)
                posts.append(r_belief_post)
                divs.append(entropy(r_belief_post, r_belief_prior))
                # we don't want KL divergence, we want the one that puts the highest probability on human's most likely goal
            # pick the goal with the lowest divergence
            # goal_idx = np.argmin(divs)
            # pick the goal that puts highest probability on goal 0
            # goal_idx = np.argmax([p[0] for p in posts])
            # pick goal with highest probability on human's most likely goal
            goal_idx = np.argmax([p[np.argmax(r_belief_prior)] for p in posts])
            # goal_idx = np.argmax([p[h_goal_idx] for p in posts])
            # goal_idx = np.argmax([p[np.argmax(r_belief_nominal.belief)] for p in posts])
            # picks the goal that human is least likely to go towards
            # goal_idx = np.argmin([p[np.argmin(r_belief_prior)] for p in posts])
            robot.goal = goals[:,[goal_idx]]
            # update robot's belief
            r_belief_post = posts[goal_idx]
            r_belief.belief = r_belief_post
            # r_belief.belief = r_belief_prior
        else:
            r_belief.belief = r_belief_prior

        # step dynamics forward
        xh0 = human.step(uh)
        xr0 = robot.step(ur)

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))
        h_beliefs = np.vstack((h_beliefs, h_belief.belief))
        r_beliefs = np.vstack((r_beliefs, r_belief.belief))
        r_beliefs_nominal = np.vstack((r_beliefs_nominal, r_belief_nominal.belief))
        r_beliefs_beta = np.dstack((r_beliefs_beta, r_belief_beta.belief))
        r_belief_likelihoods.append(likelihoods)
        # save human's actual intended goal
        h_goal_idxs.append(np.argmin(np.linalg.norm(human.goal - goals, axis=0)))
        # save robot's actual intended goal
        r_goal_idxs.append(np.argmin(np.linalg.norm(robot.goal - goals, axis=0)))

        # plot
        ax.clear()
        overlay_timesteps(ax, xh_traj, xr_traj, n_steps=idx)
        ax.scatter(xh0[0], xh0[2], c="blue")
        ax.scatter(xr0[0], xr0[2], c="red")
        ax.scatter(goals[0], goals[2], c=goal_colors)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        h_belief_ax.clear()
        h_belief_ax.plot(h_beliefs[:,0], label="P(g0)", c=goal_colors[0])
        h_belief_ax.plot(h_beliefs[:,1], label="P(g1)", c=goal_colors[1])
        h_belief_ax.plot(h_beliefs[:,2], label="P(g2)", c=goal_colors[2])
        h_belief_ax.set_xlabel("h belief of r")
        h_belief_ax.legend()

        r_belief_ax.clear()
        r_belief_ax.plot(r_beliefs[:,0], label="P(g0)", c=goal_colors[0])
        r_belief_ax.plot(r_beliefs[:,1], label="P(g1)", c=goal_colors[1])
        r_belief_ax.plot(r_beliefs[:,2], label="P(g2)", c=goal_colors[2])
        # plot nomninal belief with dashed lines
        r_belief_ax.plot(r_beliefs_nominal[:,0], c=goal_colors[0], linestyle="--")
        r_belief_ax.plot(r_beliefs_nominal[:,1], c=goal_colors[1], linestyle="--")
        r_belief_ax.plot(r_beliefs_nominal[:,2], c=goal_colors[2], linestyle="--")
        r_belief_ax.set_xlabel("r belief of h")
        r_belief_ax.legend()

        h_goal_ax.clear()
        h_goal_ax.plot(h_goal_idxs, c="blue", label="h goal")
        h_goal_ax.plot(r_goal_idxs, c="red", label="r goal")
        h_goal_ax.legend()

        r_beta_ax.clear()
        # for now, plotting marginalized belief
        for beta_idx in range(r_belief_beta.betas.shape[0]):
            r_beta_ax.plot(r_beliefs_beta[:,beta_idx,:].sum(axis=0), label=f"b={r_belief_beta.betas[beta_idx]}")
        r_beta_ax.legend()

        r_likelihoods_ax.clear()
        l = np.array(r_belief_likelihoods)
        for theta_idx in range(r_belief_nominal.thetas.shape[1]):
            r_likelihoods_ax.plot(l[:,theta_idx], label=f"theta={theta_idx}", c=goal_colors[theta_idx])
        r_likelihoods_ax.legend()

        plt.pause(0.01)
    plt.show()

if __name__ == "__main__":
    plot_rollout()
