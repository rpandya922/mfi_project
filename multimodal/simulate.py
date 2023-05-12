import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dynamics import Unicycle, LTI

def overlay_timesteps(ax, xh_traj, xr_traj, goals=None, n_steps=100, h_cmap="Blues", r_cmap="Reds", linewidth=2):
    
    if len(xh_traj) > 0:
        # human trajectory
        points = xh_traj[[0,2],:].T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        n_steps = xh_traj.shape[1]
        norm = plt.Normalize(0, n_steps)
        lc = LineCollection(segments, cmap=h_cmap, norm=norm, linewidth=linewidth)
        # Set the values used for colormapping
        lc.set_array(np.arange(n_steps+1))
        line = ax.add_collection(lc)
        # fig.colorbar(line, ax=ax)

    # robot trajectory
    if len(xr_traj) > 0:
        points = xr_traj[[0,1],:].T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        n_steps = xr_traj.shape[1]
        norm = plt.Normalize(0, n_steps)
        lc = LineCollection(segments, cmap=r_cmap, norm=norm, linewidth=linewidth)
        # Set the values used for colormapping
        lc.set_array(np.arange(n_steps+1))
        line = ax.add_collection(lc)

    if goals:
        ax.scatter(goals[0], goals[2], c=['#3A637B', '#C4A46B', '#FF5A00'])

def test_unicycle_goal_reach():
    # W = np.diag([0.05, 0.05, 0.05, 0.05])
    W = np.diag([0, 0, 0, 0])
    dyn = Unicycle(0.1, W=W)

    x0 = np.array([[0, 0, 0, 0]]).T
    goal = np.array([[-4, 3]]).T
    T = 10 # in seconds
    N = int(T / dyn.ts)

    fig, ax = plt.subplots()
    xr_traj = x0
    ur_traj = np.zeros((2, 1))
    # simulate for T seconds
    for i in range(N):
        u = dyn.compute_goal_control(x0, goal)
        x0 = dyn.step(x0, u)

        # save data
        xr_traj = np.hstack((xr_traj, x0))
        ur_traj = np.hstack((ur_traj, u))

        ax.cla()
        overlay_timesteps(ax, [], xr_traj, n_steps=i+1)
        ax.scatter(x0[0], x0[1], c="red")
        ax.scatter(goal[0], goal[1], c="green")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        plt.pause(0.01)

    plt.figure()
    plt.plot(xr_traj[0], label="x")
    plt.plot(xr_traj[1], label="y")
    plt.plot(xr_traj[2], label="v")
    plt.plot(xr_traj[3], label="psi (heading)")
    # plot goal as horizontal line
    plt.plot(np.ones(N) * goal[0], label="goal_x")
    plt.plot(np.ones(N) * goal[1], label="goal_y")
    plt.legend()

    plt.figure()
    plt.plot(ur_traj[0], label="v_dot")
    plt.plot(ur_traj[1], label="psi_dot")
    plt.legend()
    
    plt.show()

def test_lti():
    W = np.diag([0.3, 0.0, 0.3, 0.0])
    # W = np.diag([0, 0, 0, 0])
    dyn = LTI(0.1, W=W)

    x0 = np.array([[0, 0, 0, 0]]).T

    # randomly initialize 3 goals
    goals = np.random.uniform(-5, 5, (2, 3))
    # goal = np.array([[-4, 0, 3, 0]]).T
    robot_x = np.array([[-2, 2]]).T

    T = 10 # in seconds
    N = int(T / dyn.ts)

    fig, ax = plt.subplots()
    xh_traj = x0
    ur_traj = np.zeros((2, 1))
    # simulate for T seconds
    for i in range(N):
        # u = dyn.compute_control(x0, goal, robot_x)
        u = dyn.compute_control(x0, goal)
        x0 = dyn.step(x0, u)

        # save data
        xh_traj = np.hstack((xh_traj, x0))
        ur_traj = np.hstack((ur_traj, u))

        ax.cla()
        overlay_timesteps(ax, xh_traj, [], n_steps=i+1)
        ax.scatter(x0[0], x0[2], c="blue")
        ax.scatter(robot_x[0], robot_x[1], c="red")
        ax.scatter(goal[0], goal[2], c="green")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)

    plt.figure()
    plt.plot(xh_traj[0], label="x")
    plt.plot(xh_traj[1], label="xdot")
    plt.plot(xh_traj[2], label="y")
    plt.plot(xh_traj[3], label="ydot")
    # plot goal as horizontal line
    plt.plot(np.ones(N) * goal[0], label="goal_x")
    plt.plot(np.ones(N) * goal[2], label="goal_y")
    plt.legend()

    plt.figure()
    plt.plot(ur_traj[0], label="xddot")
    plt.plot(ur_traj[1], label="yddot")
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    # test_unicycle_goal_reach()
    test_lti()
