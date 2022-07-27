import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import torch
softmax = torch.nn.Softmax(dim=1)

from dynamics import DIDynamics
from human import Human, RuleBasedHuman
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from intention_utils import initialize_problem, overlay_timesteps, process_model_input
from intention_predictor import create_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def static_influence_plot():
    xh0 = np.zeros(4)
    dmin = 3.0
    ts = 0.05

    goals = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]).T
    h_goal = goals[:,[0]]

    h_dynamics = DIDynamics(ts=ts)
    human = Human(xh0, h_dynamics, goals, h_goal, gamma=350)

    xs = np.arange(-10, 10, 0.2)
    ys = np.arange(-10, 10, 0.2)
    zs = np.zeros((xs.shape[0], ys.shape[0]))

    fig, ax = plt.subplots()

    all_x = []
    all_y = []
    colors = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            xr = np.array([[x, 0.0, y, 0.0]]).T
            uh = human.get_u(xr)
            xh_next = human.dynamics.step(human.x, uh)
            diff = np.linalg.norm(xh0-xh_next)
            zs[i, j] = diff

            all_x.append(x)
            all_y.append(y)
            colors.append(diff)
    print(np.amax(colors))
    print(np.amin(colors))
    # safety margin
    ell = Ellipse(xy=(xh0[0],xh0[2]),
                  width=dmin*2, height=dmin*2,
                  fill=None, color="r")
    ax.scatter(all_x, all_y, c=colors, cmap="inferno", vmax=248.721, vmin=0.0)
    ax.add_patch(ell)
    ax.scatter([-0.1], [-0.1])
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect('equal', adjustable='box')

if __name__ == "__main__":
    horizon = 100
    n_initial = 0
    n_future = 25
    u_max = 20
    ts = 0.05
    k_plan = 20
    k_hist = 5

    model = create_model(horizon_len=k_plan)
    # model.load_state_dict(torch.load("./data/models/sim_intention_predictor_plan20.pt", map_location=device))
    model.load_state_dict(torch.load("./data/models/sim_intention_predictor_bayes.pt", map_location=device))
    # model.load_state_dict(torch.load("./data/models/sim_intention_predictor_rule.pt", map_location=device))
    model.eval()
    torch.manual_seed(1)

    np.random.seed(1)
    # human, robot, goals = initialize_problem()
    
    # creating human and robot
    xh0 = np.array([[0, 0.0, -5, 0.0]]).T
    xr0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T

    goals = np.array([
        [5.0, 0.0, 0.0, 0.0],
        [-5.0, 0.0, 5.0, 0.0],
        [5.0, 0.0, 5.0, 0.0],
    ]).T
    r_goal = goals[:,[0]]

    # xh0 = np.random.uniform(size=(4, 1))*20 - 10
    # xh0[[1,3]] = np.zeros((2, 1))
    # xr0 = np.random.uniform(size=(4, 1))*20 - 10
    # xr0[[1,3]] = np.zeros((2, 1))
    # goals = np.random.uniform(size=(4, 3))*20 - 10
    # goals[[1,3],:] = np.zeros((2, 3))
    # r_goal = goals[:,[np.random.randint(0,3)]]

    h_dynamics = DIDynamics(ts=ts)
    r_dynamics = DIDynamics(ts=ts)

    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=1)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=1)
    # human = Human(xh0, h_dynamics, goals)
    # human = RuleBasedHuman(xh0, h_dynamics, goals)

    robot = Robot(xr0, r_dynamics, r_goal, dmin=3)
    robot.set_goals(goals)

    xh_traj = np.zeros((4, horizon))
    xr_traj = np.zeros((4, horizon))

    fig, ax = plt.subplots()

    for i in range(n_initial):
         # save data
        xh_traj[:,[i]] = human.x
        xr_traj[:,[i]] = robot.x
        
         # take step
        uh = human.get_u(robot.x)
        if i == 0:
            ur = robot.get_u(human.x, robot.x, human.x)
        else:
            ur = robot.get_u(human.x, xr_traj[:,[i-1]], xh_traj[:,[i-1]])

        xh = human.step(uh)
        xr = robot.step(ur)

    # overlay_timesteps(ax, xh_traj[:,:n_initial], xr_traj[:,:n_initial], goals, n_steps=n_initial)

    # consider one possible future
    human2 = human.copy()
    robot2 = robot.copy()
    for i in range(n_initial, n_initial+n_future):
        # save data
        xh_traj[:,[i]] = human2.x
        xr_traj[:,[i]] = robot2.x
        
        # take step
        uh = human2.get_u(robot2.x)
        ur = robot2.get_u(human2.x, xr_traj[:,[i-1]], xh_traj[:,[i-1]])

        xh = human2.step(uh)
        xr = robot2.step(ur)

    # overlay_timesteps(ax, xh_traj[:,n_initial:n_initial+n_future], xr_traj[:,n_initial:n_initial+n_future], 
    #     goals, n_steps=n_future)

    cmaps = ["hot_r", "Purples", "Blues", "YlGnBu", "Greens", "Reds", "Oranges", "Greys", "summer_r", "cool_r"]
    xh_traj_to_plot = []
    xr_traj_to_plot = []
    entropies = []
    goal_probs = []
    ur_traj = []
    for idx in range(50):
        # consider another possible future
        human2 = human.copy()
        robot2 = robot.copy()
        r_cmap = "Reds" # cmaps[idx % len(cmaps)]
        h_cmap = "Blues" # cmaps[idx % len(cmaps)]
        ur_traj.append([])
        for i in range(n_initial, n_initial+n_future):
            # save data
            xh_traj[:,[i]] = human2.x
            xr_traj[:,[i]] = robot2.x
            
            # take step
            uh = human2.get_u(robot2.x)
            ur = np.random.uniform(low=-u_max, high=u_max, size=(robot2.dynamics.m,1))

            # update human's belief (if applicable)
            if type(human2) == BayesHuman:
                human2.update_belief(robot2.x, ur)

            xh = human2.step(uh)
            xr = robot2.step(ur)

            # save controls
            ur_traj[-1].append(ur)

        # get the predicted intention of the human based on this trajectory
        xh_hist = xh_traj[:,n_initial:n_initial+k_hist]
        xr_hist = xr_traj[:,n_initial:n_initial+k_hist]
        xr_plan = xr_traj[:,n_initial+k_hist:n_initial+k_hist+k_plan]

        probs = softmax(model(*process_model_input(xh_hist, xr_hist, xr_plan.T, goals)))
        probs = probs + 1e-10
        logits = torch.log2(probs)
        entropy = -torch.sum(probs * logits).item()

        xh_traj_to_plot.append(np.copy(xh_traj[:,n_initial:n_initial+n_future]))
        xr_traj_to_plot.append(np.copy(xr_traj[:,n_initial:n_initial+n_future]))
        entropies.append(entropy)
        goal_probs.append(probs.detach().numpy())

        # overlay_timesteps(ax, xh_traj[:,n_initial:n_initial+n_future], xr_traj[:,n_initial:n_initial+n_future], 
        #     goals, n_steps=n_future, h_cmap=h_cmap, r_cmap=r_cmap)

    entropies = np.array(entropies)

    # sort by entropy values 
    idx = np.argsort(entropies)
    xh_traj_to_plot = [xh_traj_to_plot[i] for i in idx]
    xr_traj_to_plot = [xr_traj_to_plot[i] for i in idx]
    goal_probs = [goal_probs[i] for i in idx]
    entropies = entropies[idx]

    # save sorted robot controls to npz file
    ur_traj_to_save = [ur_traj[i] for i in idx]
    np.savez("./data/ur_traj_sorted.npz", ur=ur_traj_to_save)

    print(goal_probs[0], goal_probs[-1])

    ent_colors = np.repeat(entropies[:,None], k_hist+k_plan, axis=1).flatten()
    xh_traj_to_plot = np.hstack(xh_traj_to_plot)
    xr_traj_to_plot = np.hstack(xr_traj_to_plot)
    ax.scatter(xr_traj_to_plot[0,:], xr_traj_to_plot[2,:], c=ent_colors, cmap="Reds", 
        s=5, vmin=np.amin(entropies)*0.8, vmax=np.amax(entropies))
    ax.scatter(xh_traj_to_plot[0,:], xh_traj_to_plot[2,:], c=ent_colors, cmap="Blues", 
        s=5, vmin=np.amin(entropies)*0.8, vmax=np.amax(entropies))
    plt.colorbar(cm.ScalarMappable(cmap="Reds"), ax=ax)
    plt.colorbar(cm.ScalarMappable(cmap="Blues"), ax=ax)
    print(np.amin(entropies), np.amax(entropies))

    # plot goals at the end
    ax.scatter(goals[0,:], goals[2,:], c=['#3A637B', '#C4A46B', '#FF5A00'])
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.show()
