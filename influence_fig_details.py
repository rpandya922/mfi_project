import numpy as np
import torch
import matplotlib.pyplot as plt
softmax = torch.nn.Softmax(dim=1)

from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from dynamics import DIDynamics
from intention_predictor import create_model
from intention_utils import rollout_control, get_robot_plan, process_model_input

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    horizon = 100
    n_initial = 0
    n_future = 25
    u_max = 20
    ts = 0.05
    k_plan = 20
    k_hist = 5

    model = create_model(horizon_len=k_plan)
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor_bayes.pt", map_location=device))
    model.eval()
    torch.manual_seed(1)

    np.random.seed(1)
    
    # creating human and robot
    xh0 = np.array([[0, 0.0, -5, 0.0]]).T
    xr0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T

    goals = np.array([
        [5.0, 0.0, 0.0, 0.0],
        [-5.0, 0.0, 5.0, 0.0],
        [5.0, 0.0, 5.0, 0.0],
    ]).T
    r_goal = goals[:,[0]]

    h_dynamics = DIDynamics(ts=ts)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=1)

    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    robot.set_goals(goals)

    ur_sorted = np.load("./data/ur_traj_sorted.npz")["ur"]
    n_traj = len(ur_sorted)

    fig, ax = plt.subplots()

    # use the highest-entropy trajectories
    for i in range(n_traj-1, n_traj-2, -1):
        ur_samples = ur_sorted[i].squeeze().T

        # initialize human and robot
        human2 = human.copy()
        robot2 = robot.copy()

        # arrays to store the results
        xh_traj = np.zeros((4, n_future+1))
        xr_traj = np.zeros((4, n_future+1))
        xh_traj[:,[0]] = human2.x
        xr_traj[:,[0]] = robot2.x
        ur_traj = np.zeros((2, n_future))
        uh_traj = np.zeros((2, n_future))

        # store the belief of the robot over time & the human's true goal over time
        r_beliefs = []
        h_goals = []

        for t in range(n_future):

            # compute the robot's prediction of the human's goal
            if t >= k_hist:
                xh_hist = xh_traj[:,t-k_hist:t]
                xr_hist = xr_traj[:,t-k_hist:t]

                # compute the robot's future plan: if t+k_plan is beyond the horizon, compute a goal-oriented plan for the remaining time
                if t+k_plan > n_future:
                    ur_plan = ur_samples[:,t:n_future]
                    # roll out saved trajectory as far as we can
                    xr_plan = rollout_control(robot2, ur_plan)
                    # compute the robot's goal-oriented plan
                    xr_goal_plan = get_robot_plan(robot2, t+k_plan-n_future, xr0=xr_plan[:,[-1]])
                    full_xr_plan = np.hstack((xr_plan, xr_goal_plan))
                else:    
                    ur_plan = ur_samples[:,t:t+k_plan]
                    full_xr_plan = rollout_control(robot2, ur_plan)

                # compute the robot's belief of the human's goal
                # TODO: find out why this belief does not match the belief output by influence_figure.py initially
                r_belief = softmax(model(*process_model_input(xh_hist, xr_hist, full_xr_plan.T, goals)))
                import ipdb; ipdb.set_trace()

            # compute controls
            ur = ur_samples[:,[t]]
            uh = human2.get_u(robot2.x)

            # update human belief (if applicable)
            if type(human2) == BayesHuman:
                human2.update_belief(robot2.x, ur)

            # take actions
            robot2.step(ur)
            human2.step(uh)

            # save data
            xh_traj[:,[t+1]] = human2.x
            xr_traj[:,[t+1]] = robot2.x
            ur_traj[:,[t]] = ur
            uh_traj[:,[t]] = uh

        # plot
        ax.cla()
        ax.plot(xh_traj[0,:], xh_traj[2,:], c="b", label="human")
        ax.plot(xr_traj[0,:], xr_traj[2,:], c="r", label="robot")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        plt.pause(0.01)
        import ipdb; ipdb.set_trace()
        input("Press Enter to continue...")
