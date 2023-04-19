import numpy as np
import matplotlib.pyplot as plt
import torch
softmax = torch.nn.Softmax(dim=1)
from tqdm import tqdm
import pickle
import os

from intention_predictor import create_model
from dynamics import DIDynamics
from bayes_inf import BayesEstimator, BayesHuman
from human import Human
from robot import Robot
from intention_utils import overlay_timesteps, get_robot_plan, process_model_input

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_trajectory(human, robot, goals, horizon=None, model=None, plot=True, robot_obs=None):
    """
    inputs:
        human: Human object
        robot: Robot object
        goals: 3xN array of goals
        horizon: number of timesteps to run for (if None, run until human reaches goal)
        model: model to use for predicting human intention
        plot: whether to plot the trajectory
        robot_obs: optional synthetic obstacle for the robot to avoid
    outputs:
        xh_traj: 3xT array of human states
        xr_traj: 3xT array of robot states
        all_r_beliefs: 3x3xT array of robot beliefs about human intention
        h_goal_reached: index of goal reached by human
    """
    # show both agent trajectories when the robot reasons about its effect on the human
    if plot:
        fig, ax = plt.subplots()
    xh = human.x
    xr = robot.x

    # if no horizon is given, run until the human reaches the goal (but set arrays to 300 and expand arrays if necessary)
    if horizon is None:
        fixed_horizon = False
        horizon = np.inf
        arr_size = 20
    else:
        fixed_horizon = True
        arr_size = horizon

    # save robot predictions if model is passed in
    if model is not None:
        all_r_beliefs = np.zeros((goals.shape[1], goals.shape[1], arr_size))
    else:
        all_r_beliefs = None
    xh_traj = np.zeros((human.dynamics.n, arr_size+1))
    xr_traj = np.zeros((robot.dynamics.n, arr_size+1))
    xh_traj[:,[0]] = xh
    xr_traj[:,[0]] = xr
    h_goal_reached = -1

    i = 0
    while i < horizon:
        if plot:
            # plotting
            ax.cla()
            # plot trajectory trail so far
            overlay_timesteps(ax, xh_traj[:,0:i], xr_traj[:,0:i], n_steps=i)

            # TODO: highlight robot's predicted goal of the human
            ax.scatter(goals[0], goals[2], c="green", s=100)
            ax.scatter(human.x[0], human.x[2], c="#034483", s=100)
            ax.scatter(robot.x[0], robot.x[2], c="#800E0E", s=100)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            # img_path = f"./data/videos/mfi_demo/frames3"
            # img_path += f"/{i:03d}.png"
            # plt.savefig(img_path, dpi=300)
            plt.pause(0.01)

        # compute the robot's prediction of the human's intention
        # but only if model is passed in
        if model is not None:
            if i < 5:
                nominal_goal_idx = 1
            else:
                xh_hist = xh_traj[:,i-5:i]
                xr_hist = xr_traj[:,i-5:i]
                r_beliefs = []
                for goal_idx in range(robot.goals.shape[1]):
                    goal = robot.goals[:,[goal_idx]]
                    # compute robot plan given this goal
                    xr_plan = get_robot_plan(robot, horizon=20, goal=goal)

                    r_beliefs.append(softmax(model(*process_model_input(xh_hist, xr_hist, xr_plan.T, goals))).detach().numpy()[0])
                # use robot's belief to pick which goal to move toward by picking the one that puts the highest probability on the human's nominal goal (what we observe in the first 5 timesteps)
                r_beliefs = np.array(r_beliefs)
                r_goal_idx = np.argmax(r_beliefs[:,nominal_goal_idx])
                # TODO: set goal only if specified
                # robot.set_goal(robot.goals[:,[r_goal_idx]])
                all_r_beliefs[:,:,i] = r_beliefs

        # compute agent controls
        uh = human.get_u(robot.x)
        ur = robot.dynamics.get_goal_control(robot.x, robot.goal)
        if robot_obs is not None:
            ur += robot.dynamics.get_robot_control(robot.x, robot_obs)
        # robot should stay still for 5 timesteps
        if i < 5:
            ur = np.zeros(ur.shape)

        # update human belief (if applicable)
        if type(human) == BayesHuman and i >=5:
            human.update_belief(robot.x, ur)

        # compute new states
        xh = human.step(uh)
        xr = robot.step(ur)

        # save data
        xh_traj[:,[i+1]] = xh
        xr_traj[:,[i+1]] = xr

        # check if human has reached a goal
        if not fixed_horizon:
            # check if we need to expand arrays
            if i >= arr_size-1:
                arr_size *= 2
                xh_traj = np.hstack((xh_traj, np.zeros((human.dynamics.n, arr_size))))
                xr_traj = np.hstack((xr_traj, np.zeros((robot.dynamics.n, arr_size))))
                if all_r_beliefs is not None:
                    all_r_beliefs = np.dstack((all_r_beliefs, np.zeros((goals.shape[1], goals.shape[1], arr_size))))
                # print("expanding arrays")
            dists = np.linalg.norm(goals[[0,2]] - xh[[0,2]], axis=0)
            if np.min(dists) < 0.5:
                h_goal_reached = np.argmin(dists)
                break
        
        # NOTE: don't put any code in the loop after this
        i += 1
    
    # return only non-zero elements of arrays
    xh_traj = xh_traj[:,0:i+1]
    xr_traj = xr_traj[:,0:i+1]
    if all_r_beliefs is not None:
        all_r_beliefs = all_r_beliefs[:,:,0:i+1]
    return xh_traj, xr_traj, all_r_beliefs, h_goal_reached

def simulate_init_cond(xr0, xh0, human, robot, goals, n_traj=10):
    """
    Runs a number of simulations from the same initial conditions and returns the probability of each goal being the human's intention
    """
    # save data
    all_xh_traj = []
    all_xr_traj = []
    all_goals_reached = []
    goals_reached = np.zeros(goals.shape[1])
    for i in range(n_traj):
        # set the human and robot's initial states
        human.x = xh0
        robot.x = xr0
        # randomly set the robot's goal to one of the options
        robot.set_goal(goals[:,[np.random.randint(0, goals.shape[1])]])
        # generate a synthetic obstacle for the robot that lies between the robotand its goal
        # get bounds of the rectangle that contains the robot and its goal
        x_min = min(robot.x[0], robot.goal[0])
        x_max = max(robot.x[0], robot.goal[0])
        y_min = min(robot.x[2], robot.goal[2])
        y_max = max(robot.x[2], robot.goal[2])
        # randomly sample a point in this rectangle
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        robot_obs = np.array([[x[0], 0, y[0], 0]]).T
        # run the simulation
        xh_traj, xr_traj, _, h_goal_reached = run_trajectory(human, robot, goals, plot=False, robot_obs=robot_obs)
        # increment the count for the goal that was reached
        goals_reached[h_goal_reached] += 1
        # save data
        all_xh_traj.append(xh_traj)
        all_xr_traj.append(xr_traj)
        all_goals_reached.append(h_goal_reached)
    # return the probability of each goal being the human's intention
    return all_xh_traj, all_xr_traj, all_goals_reached, goals_reached / n_traj, goals

def create_labels(all_xh_traj, all_xr_traj, all_goals_reached, goal_probs, goals, mode="interpolate", history=5, horizon=5):

    input_traj = []
    robot_future = []
    input_goals = []
    labels = []

    for i in range(len(all_xh_traj)):
        goal_prob_label = goal_probs.copy()
        xh_traj = all_xh_traj[i]
        xr_traj = all_xr_traj[i]
        goal_reached = all_goals_reached[i]
        
        prob_step_sizes = -goal_prob_label / xh_traj.shape[1]
        prob_step_sizes[goal_reached] = (1 - goal_prob_label[goal_reached]) / xh_traj.shape[1]
        for j in range(xh_traj.shape[1]):
            # set the label for the current timestep
            if mode == "interpolate":
                goal_prob_label += prob_step_sizes

            # zero-pad inputs if necessary
            xh_hist = np.zeros((xh_traj.shape[0], history))
            xr_hist = np.zeros((xr_traj.shape[0], history))
            xr_future = np.zeros((xr_traj.shape[0], horizon))
            
            if j < history:
                xh_hist[:,history-(j+1):] = xh_traj[:,0:j+1]
                xr_hist[:,history-(j+1):] = xr_traj[:,0:j+1]
            else:
                xh_hist = xh_traj[:,(j+1)-history:j+1]
                xr_hist = xr_traj[:,(j+1)-history:j+1]

            if j + horizon >= xh_traj.shape[1]:
                xr_future[:,0:xh_traj.shape[1]-(j+1)] = xr_traj[:,j+1:]
            else:
                xr_future = xr_traj[:,j+1:(j+1)+horizon]

            # add data point to dataset
            # LSTM expects input of size (sequence length, # features) [batch size dealth with separately]
            input_traj.append(torch.tensor(np.vstack((xh_hist, xr_hist)).T).float().to(device)) # shape (5,8)
            robot_future.append(torch.tensor(xr_future.T).float().to(device)) # shape (5,4)
            input_goals.append(torch.tensor(goals).float().to(device)) # shape (4,3)
            labels.append(torch.tensor(goal_prob_label).float().to(device)) # shape (3,)

            # add a data point that has xr_future zero'd out and uses initial goal probs as labels
            input_traj.append(torch.tensor(np.vstack((xh_hist, xr_hist)).T).float().to(device)) # shape (5,8)
            robot_future.append(torch.tensor(np.zeros_like(xr_future).T).float().to(device)) # shape (5,4)
            input_goals.append(torch.tensor(goals).float().to(device)) # shape (4,3)
            labels.append(torch.tensor(goal_probs).float().to(device)) # shape (3,)

    return input_traj, robot_future, input_goals, labels

def create_dataset(n_init_cond=200):
    # generate n_init_cond initial conditions and find goal_probs for each

    # saving raw data
    xh_traj = []
    xr_traj = []
    goals_reached = []
    goal_probs = []
    all_goals = []
    for i in tqdm(range(n_init_cond)):
        # generate initial conditions
        ts = 0.05
        xh0 = np.random.uniform(-10, 10, (4, 1))
        xh0[[1,3]] = 0
        xr0 = np.random.uniform(-10, 10, (4, 1))
        xr0[[1,3]] = 0
        goals = np.random.uniform(-10, 10, (4, 3))
        goals[[1,3]] = 0
        r_goal = goals[:,[0]] # this is arbitrary since it'll be changed in simulations later anyways

        # create human and robot objects
        W = np.diag([0.0, 0.7, 0.0, 0.7])
        h_dynamics = DIDynamics(ts=ts, W=W)
        r_dynamics = DIDynamics(ts=ts)
        r_dynamics.gamma = 10 # for synthetic obstacles for the robot

        belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
        human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=5)
        # human = Human(xh0, h_dynamics, goals, gamma=5)
        robot = Robot(xr0, r_dynamics, r_goal, dmin=3)

        # simulate trajectories
        data = simulate_init_cond(xr0, xh0, human, robot, goals, n_traj=100)
        xh_traj.append(data[0])
        xr_traj.append(data[1])
        goals_reached.append(data[2])
        goal_probs.append(data[3])
        all_goals.append(data[4])

    return xh_traj, xr_traj, goals_reached, goal_probs, all_goals

def save_data(dataset, path="./data/simulated_interactions_bayes_prob_train.pkl"):
    # check if path exists
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump({"xh_traj": dataset[0], "xr_traj": dataset[1], "goals_reached": dataset[2], "goal_probs": dataset[3], "goals": dataset[4]}, f)

def process_and_save_data(raw_data_path, processed_data_path):
    raw_data = pickle.load(open(raw_data_path, "rb"))
    xh_traj = raw_data["xh_traj"]
    xr_traj = raw_data["xr_traj"]
    goals_reached = raw_data["goals_reached"]
    goal_probs = raw_data["goal_probs"]
    goals = raw_data["goals"]

    input_traj = []
    robot_future = []
    input_goals = []
    labels = []

    n_traj = len(xh_traj)
    for i in range(n_traj):
        it, rf, ig, l = create_labels(xh_traj[i], xr_traj[i], goals_reached[i], goal_probs[i], goals[i])
        input_traj += it
        robot_future += rf
        input_goals += ig
        labels += l

    # save in pkl file
    if not os.path.exists(os.path.dirname(processed_data_path)):
        os.makedirs(os.path.dirname(processed_data_path))
    with open(processed_data_path, "wb") as f:
        pickle.dump({"input_traj": torch.stack(input_traj), "robot_future": torch.stack(robot_future), "input_goals": torch.stack(input_goals), "labels": torch.stack(labels)}, f)

if __name__ == "__main__":
    raw_data_path = "./data/simulated_interactions_bayes_prob_train.pkl"
    processed_data_path = "./data/simulated_interactions_bayes_prob_train_processed.pkl"
    dataset = create_dataset(n_init_cond=1000)
    save_data(dataset)
    process_and_save_data(raw_data_path, processed_data_path)