# NN-based conditional behavior prediction
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import torch
softmax = torch.nn.Softmax(dim=1)
import pickle

from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman  
from cbp_model import CBPEstimator, BetaBayesEstimator
from robot import Robot
from intention_utils import overlay_timesteps, get_robot_plan, process_model_input
from intention_predictor import create_model
from nn_prob_pred import compute_features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_trained_model(model_path, horizon=20, hidden_size=128, num_layers=2, hist_feats=8, plan_feats=4, stats_file=None):
    # load model
    model = create_model(horizon_len=horizon, hidden_size=hidden_size, num_layers=num_layers, hist_feats=hist_feats, plan_feats=plan_feats)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    if stats_file is not None:
        with open(stats_file, "rb") as f:
            stats = pickle.load(f)
    else:
        stats = None

    return model, stats

def compute_post_nn(model, stats, robot, xh_hist, xr_hist, r_goal, goals, k_plan=20, feats=True):
    xr_plan = get_robot_plan(robot, horizon=k_plan, goal=r_goal)
    if feats:
        hist_feats, future_feats = compute_features(xh_hist, xr_hist, xr_plan, goals)
        input_hist, input_future, input_goals = process_model_input(xh_hist, xr_hist, xr_plan.T, goals)
        input_hist = torch.cat((input_hist, torch.tensor(hist_feats.T).float().unsqueeze(0)), dim=2)
        input_future = torch.cat((input_future, torch.tensor(future_feats.T).float().unsqueeze(0)), dim=2)
        # normalize features with same mean and std as training data
        if stats is not None:
            input_hist = (input_hist - stats["input_traj_mean"]) / stats["input_traj_std"]
            input_future = (input_future - stats["robot_future_mean"]) / stats["robot_future_std"]
            input_goals = (input_goals.transpose(1,2) - stats["input_goals_mean"]) / stats["input_goals_std"]
            input_goals = input_goals.transpose(1,2)
        model_out = model(input_hist, input_future, input_goals)
    else:
        model_out = model(*process_model_input(xh_hist, xr_hist, xr_plan.T, goals))

    return softmax(model_out).detach().numpy()[0]

def plot_rollout(model_path, stats_file, hist_feats=21, plan_feats=10, k_hist=5, k_plan=20, feats=True):
    # load trained model
    model, stats = load_trained_model(model_path, hist_feats=hist_feats, plan_feats=plan_feats, stats_file=stats_file)
    model.eval()

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
    r_belief = np.ones(goals.shape[1]) / goals.shape[1]
    r_belief_nominal = CBPEstimator(thetas=goals, dynamics=h_dynamics, beta=2)
    r_belief_beta = BetaBayesEstimator(thetas=goals, betas=[0.01, 0.1, 1, 10, 100, 1000], dynamics=h_dynamics)

    # data saving
    xh_traj = xh0
    xr_traj = xr0
    # simulate for T seconds
    N = int(7.5 / ts)
    h_beliefs = h_belief.belief
    r_beliefs = r_belief
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
        uh = human.get_u(robot.x)
        # uh = np.zeros((2,1)) #+ np.random.uniform(-5, 5, (2,1))
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)

        # update human's belief
        human.update_belief(robot.x, ur)
        # simulate robot nominal belief update
        r_belief_nominal.belief, likelihoods = r_belief_nominal.update_belief(human.x, uh, robot.x, return_likelihood=True)
        r_belief_prior = r_belief_nominal.belief
        
        # update the robot's belief over goals and betas
        r_belief_beta2 = r_belief_beta.update_belief_(human.x, uh, robot.x)
        r_belief_beta.belief = r_belief_beta2

        if idx > 5:
            # loop through goals and compute belief update for each
            divs = []
            posts = []
            for goal_idx in range(goals.shape[1]):
                goal = goals[:,[goal_idx]]
                # compute CBP belief update
                r_belief_post = compute_post_nn(model, stats, robot, xh_traj[:,-k_hist:], xr_traj[:,-k_hist:], goal, goals, k_plan=k_plan, feats=feats)
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

            # update robot's goal
            robot.goal = goals[:,[goal_idx]]
            # update robot's belief
            r_belief = posts[goal_idx]

        # step dynamics forward
        xh0 = human.step(uh)
        xr0 = robot.step(ur)

        # if human reaches goal, break
        # if np.linalg.norm(xh0[[0,2]] - human.goal[[0,2]]) < 0.5:
        #     print("Human reached goal")
        #     break

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))
        h_beliefs = np.vstack((h_beliefs, h_belief.belief))
        r_beliefs = np.vstack((r_beliefs, r_belief))
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
        # highlight robot's predicted goal of the human
        if idx >= 5:
            # plot a transparent circle on each of the goals with radius proportional to the robot's belief
            for goal_idx in range(goals.shape[1]):
                goal = goals[:,[goal_idx]]
                # plot a circle with radius proportional to the robot's belief
                ax.add_artist(plt.Circle((goal[0], goal[2]), r_belief[goal_idx]*2, color=goal_colors[goal_idx], alpha=0.3))

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
    model_path = "./data/models/prob_pred_intention_predictor_bayes_20230623-12.pt"
    stats_file = "./data/models/prob_pred_intention_predictor_bayes_20230623-12_stats.pkl"
    k_hist = 5
    k_plan = 20
    plot_rollout(model_path, stats_file, k_hist=k_hist, k_plan=k_plan)
