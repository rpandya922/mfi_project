import numpy as np
import torch
softmax = torch.nn.Softmax(dim=1)
import matplotlib.pyplot as plt

from human import Human
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from dynamics import DIDynamics
from intention_predictor import create_model
from intention_utils import process_model_input, get_robot_plan

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_interaction(ts, horizon, k_hist, k_plan, model, plot_sim=False, print_pred=False):
    # randomly initialize xh0, xr0, goals
    xh0 = np.random.uniform(size=(4, 1))*20 - 10
    xh0[[1,3]] = np.zeros((2, 1))
    xr0 = np.random.uniform(size=(4, 1))*20 - 10
    xr0[[1,3]] = np.zeros((2, 1))

    goals = np.random.uniform(size=(4, 3))*20 - 10
    goals[[1,3],:] = np.zeros((2, 3))
    # check if all goals are at least 3.0 away from each other
    while np.linalg.norm(goals[:,0] - goals[:,1]) < 3.0 or np.linalg.norm(goals[:,0] - goals[:,2]) < 3.0 or np.linalg.norm(goals[:,1] - goals[:,2]) < 3.0:
        goals = np.random.uniform(size=(4, 3))*20 - 10
        goals[[1,3],:] = np.zeros((2, 3))

    r_goal = goals[:,[np.random.randint(0,3)]]

    h_dynamics = DIDynamics(ts=ts)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=1)
    # human = Human(xh0, h_dynamics, goals)
    robot = Robot(xr0, r_dynamics, r_goal)

    xh_traj = np.zeros((4, horizon))
    xr_traj = np.zeros((4, horizon))
    h_goals = np.zeros((4, horizon))
    h_goal_reached = np.zeros((1, horizon))

    if plot_sim:
        _, ax = plt.subplots()

    human_intentions = []
    robot_predictions = []
    timestep_99 = horizon+1
    correct_pred  = []

    for i in range(horizon):
        # plot human, robot, and goals
        if plot_sim:
            ax.cla()
            ax.scatter(human.x[0], human.x[2], c='xkcd:medium blue')
            ax.scatter(robot.x[0], robot.x[2], c='xkcd:scarlet')
            ax.scatter(goals[0], goals[2], c=['#3A637B', '#C4A46B', '#FF5A00'])
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            plt.pause(0.01)

        # save data
        xh_traj[:,[i]] = human.x
        xr_traj[:,[i]] = robot.x
        h_goals[:,[i]] = human.get_goal()
        # check if human reached its goal
        if np.linalg.norm(human.x - human.get_goal()) < 0.1:
            h_goal_reached[:,i] = 1

        if i > k_hist:
            # building the inputs to the model
            xh_hist = xh_traj[:,i-k_hist+1:i+1]
            xr_hist = xr_traj[:,i-k_hist+1:i+1]
            xr_plan = get_robot_plan(robot, horizon=k_plan)
            
            # print human's belief of robot's goal
            if print_pred:
                robot_i = np.argmin(np.linalg.norm(robot.goal - goals, axis=0))
                print(f"Human belief: {np.argmax(human.belief.belief)}, Robot goal: {robot_i}")
                print()

            goal_probs = softmax(model(*process_model_input(xh_hist, xr_hist, xr_plan.T, goals)))

            est_goal_idx = torch.argmax(goal_probs).item()
            _, h_goal_idx = human.get_goal(get_idx=True)

            human_intentions.append(h_goal_idx)
            robot_predictions.append(goal_probs)
            if goal_probs[0,h_goal_idx] > 0.99 and est_goal_idx == h_goal_idx:
                timestep_99 = min(timestep_99, i)
            correct_pred.append(est_goal_idx == h_goal_idx)

            if print_pred:
                print(f"Predicted goal: {est_goal_idx}, True goal: {h_goal_idx}, P(correct): {goal_probs[0,h_goal_idx].item()}")
                # print(h_goal_idx, est_goal_idx, goal_probs[0,est_goal_idx].item())

        # take step
        uh = human.get_u(robot.x)
        if i == 0:
            ur = robot.get_u(human.x, robot.x, human.x)
        else:
            ur = robot.get_u(human.x, xr_traj[:,[i-1]], xh_traj[:,[i-1]])

        # update human's belief (if applicable)
        if type(human) == BayesHuman:
            human.update_belief(robot.x, ur)

        xh = human.step(uh)
        xr = robot.step(ur)
    
    return xh_traj, xr_traj, goals, human_intentions, robot_predictions, timestep_99, correct_pred

if __name__ == "__main__":
    ts = 0.05
    horizon = 100
    k_hist = 5
    k_plan = 20

    model = create_model(horizon_len=k_plan)
    # model.load_state_dict(torch.load("./data/models/sim_intention_predictor_plan20.pt", map_location=device))
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor_bayes.pt", map_location=device))
    model.eval()
    plot_sim = True
    print_pred = True

    np.random.seed(0)
    torch.manual_seed(0)

    corrects = []
    last_corrects = []
    timestep_99s = []
    for ii in range(10):
        xh_traj, xr_traj, goals, human_intentions, robot_predictions, timestep_99, correct_pred = simulate_interaction(ts, horizon, k_hist, k_plan, model, plot_sim=plot_sim, print_pred=print_pred)
        print(np.mean(correct_pred))

        corrects.append(np.mean(correct_pred))
        last_corrects.append(correct_pred[-1])
        timestep_99s.append(timestep_99)
        print()
    print(f"Mean correct prediction: {np.mean(corrects)}")
    print(f"Last correct prediction: {np.mean(last_corrects)}")
    print(f"Mean timestep 99: {np.mean(timestep_99s)}")

