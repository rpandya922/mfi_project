import numpy as np
import torch
import matplotlib.pyplot as plt
softmax = torch.nn.Softmax(dim=1)

from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from dynamics import DIDynamics
from intention_predictor import create_model
from intention_utils import rollout_control, get_robot_plan, process_model_input, overlay_timesteps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    horizon = 100
    n_initial = 0
    n_future = 25
    horizon = 25
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

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
    axes = np.array(axes).flatten()

    # _, ax2 = plt.subplots()

    # use the highest-entropy trajectories
    # for i in range(n_traj-1, n_traj-6, -1):
    for i in range(5):
    # for i in [45]:
        ur_samples = ur_sorted[i].squeeze().T

        # initialize human and robot
        human2 = human.copy()
        robot2 = robot.copy()

        # arrays to store the results
        xh_traj = np.zeros((4, horizon+1))
        xr_traj = np.zeros((4, horizon+1))
        xh_traj[:,[0]] = human2.x
        xr_traj[:,[0]] = robot2.x
        ur_traj = np.zeros((2, horizon))
        uh_traj = np.zeros((2, horizon))

        # store the belief of the robot over time & the human's true goal over time
        r_beliefs = []
        h_beliefs = []
        h_goals = []

        h_beliefs.append(human2.belief.belief.copy())

        for t in range(horizon):
    
            # ax2.cla()
            # ax2.scatter(goals[0], goals[2], c=["#3A637B", "#C4A46B", "#FF5A00"])
            # ax2.scatter(human2.x[0], human2.x[2], c="#034483")
            # ax2.scatter(robot2.x[0], robot2.x[2], c="#800E0E")
            # ax2.set_xlim(-10, 10)
            # ax2.set_ylim(-10, 10)
            # plt.pause(0.01)

            # compute the robot's prediction of the human's goal
            if t >= k_hist-1:
                xh_hist = xh_traj[:,t-k_hist+1:t+1]
                xr_hist = xr_traj[:,t-k_hist+1:t+1]

                # compute the robot's future plan: if t+k_plan is beyond the horizon, compute a goal-oriented plan for the remaining time
                if t+k_plan > n_future:
                    ur_plan = ur_samples[:,t:n_future]
                    # roll out saved trajectory as far as we can
                    if ur_plan.shape[1] > 0:
                        xr_plan = rollout_control(robot2, ur_plan)
                        # compute the robot's goal-oriented plan
                        xr_goal_plan = get_robot_plan(robot2, t+k_plan-n_future, xr0=xr_plan[:,[-1]])
                        full_xr_plan = np.hstack((xr_plan, xr_goal_plan))
                    else:
                        full_xr_plan = get_robot_plan(robot2, 20, xr0=xr_plan[:,[-1]])
                else:    
                    ur_plan = ur_samples[:,t:t+k_plan]
                    full_xr_plan = rollout_control(robot2, ur_plan)

                # compute the robot's belief of the human's goal
                # TODO: find out why this belief does not match the belief output by influence_figure.py initially
                r_belief = softmax(model(*process_model_input(xh_hist, xr_hist, full_xr_plan.T, goals)))
                r_beliefs.append(r_belief.detach().numpy())

            # compute controls
            if t < n_future:
                ur = ur_samples[:,[t]]
            else:
                ur = robot2.dynamics.get_goal_control(robot2.x, robot2.goal)
            uh = human2.get_u(robot2.x)

            # update human belief (if applicable)
            if type(human2) == BayesHuman:
                human2.update_belief(robot2.x, ur)

            # take actions
            robot2.step(ur)
            human2.step(uh)

            # save data
            h_goals.append(human2.get_goal(get_idx=True)[1])
            h_beliefs.append(human2.belief.belief.copy())
            xh_traj[:,[t+1]] = human2.x
            xr_traj[:,[t+1]] = robot2.x
            ur_traj[:,[t]] = ur
            uh_traj[:,[t]] = uh

        # plot
        ax = axes[0]
        ax.cla()
        overlay_timesteps(ax, xh_traj, xr_traj, goals, n_steps=n_future)
        # ax.plot(xh_traj[0,:], xh_traj[2,:], c="b", label="human")
        # ax.plot(xr_traj[0,:], xr_traj[2,:], c="r", label="robot")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        ax = axes[1]
        ax.cla()
        intial_b = np.ones((k_hist, goals.shape[1])) / goals.shape[1]
        r_beliefs = np.vstack(r_beliefs)
        r_beliefs = np.vstack((intial_b, r_beliefs))
        ax.plot(r_beliefs[:,0], label="P(g0)", c="#3A637B")
        ax.plot(r_beliefs[:,1], label="P(g1)", c="#C4A46B")
        ax.plot(r_beliefs[:,2], label="P(g2)", c="#FF5A00")
        ax.set_ylim(0, 1)

        # color between vertical lines for the human's goal, color depends on the actual goal
        colors = ["#3A637B", "#C4A46B", "#FF5A00"]
        for i, g in enumerate(h_goals):
            ax.axvspan(i, i+1, color=colors[g], alpha=0.3)
        ax.legend()
        ax.set_title("Robot's belief + human's true goal")

        # plot the human's belief over time
        ax = axes[2]
        ax.cla()
        h_beliefs = np.vstack(h_beliefs)
        ax.plot(h_beliefs[:,0], label="P(g0)", c="#3A637B")
        ax.plot(h_beliefs[:,1], label="P(g1)", c="#C4A46B")
        ax.plot(h_beliefs[:,2], label="P(g2)", c="#FF5A00")
        ax.set_ylim(0, 1)
        ax.set_title("Human's belief")

        # TODO: plot the human and robot over time to see why it looks like human is going to orange goal at the end
        plt.pause(0.01)
        input("Press Enter to continue...")
