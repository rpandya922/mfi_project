import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from dynamics import DIDynamics
from human import Human
from bayes_inf import BayesEstimator, BayesHuman
from robot import Robot
from optimize_uncertainty import initialize_problem, overlay_timesteps

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

    # static_influence_plot()
    # plt.show()
    # 1/0

    horizon = 100
    n_initial = 0
    n_future = 25
    u_max = 20
    ts = 0.05

    np.random.seed(0)
    human, robot, goals = initialize_problem()
    
    # creating human and robot
    # xh0 = np.array([[3.0, 0.0, 0.0, 0.0]]).T
    xh0 = np.array([[0, 0.0, -5, 0.0]]).T
    xr0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T

    goals = np.array([
        [5.0, 0.0, 0.0, 0.0],
        [-5.0, 0.0, 5.0, 0.0],
        [5.0, 0.0, 5.0, 0.0],
    ]).T
    # h_goal = goals[:,[1]]
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

    # human = Human(xh0, h_dynamics, goals, h_goal, gamma=1)
    # human = Human(xh0, h_dynamics, goals, gamma=1)
    belief = BayesEstimator(thetas=goals, dynamics=r_dynamics, beta=20)
    human = BayesHuman(xh0, h_dynamics, goals, belief, gamma=1)

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

    overlay_timesteps(ax, xh_traj[:,:n_initial], xr_traj[:,:n_initial], goals, n_steps=n_initial)

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

    # plt.show()
    # 1/0

    overlay_timesteps(ax, xh_traj[:,n_initial:n_initial+n_future], xr_traj[:,n_initial:n_initial+n_future], 
        goals, n_steps=n_future)

    cmaps = ["hot_r", "Purples", "Blues", "YlGnBu", "Greens", "Reds", "Oranges", "Greys", "summer_r", "cool_r"]
    for idx in range(50):
        # consider another possible future
        human2 = human.copy()
        robot2 = robot.copy()
        r_cmap = "Reds" # cmaps[idx % len(cmaps)]
        h_cmap = "Blues" # cmaps[idx % len(cmaps)]
        for i in range(n_initial, n_initial+n_future):
            # plot human and robot positions
            # ax.cla()
            # ax.scatter(human2.x[0], human2.x[2], c="blue")
            # ax.scatter(robot2.x[0], robot2.x[2], c="red")
            # ax.scatter(goals[0], goals[2])
            # ax.set_xlim(-10, 10)
            # ax.set_ylim(-10, 10)
            # plt.pause(0.01)

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

        overlay_timesteps(ax, xh_traj[:,n_initial:n_initial+n_future], xr_traj[:,n_initial:n_initial+n_future], 
            goals, n_steps=n_future, h_cmap=h_cmap, r_cmap=r_cmap)

    # TODO: plot goals at the end
    plt.show()
        
