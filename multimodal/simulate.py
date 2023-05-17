import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dynamics import Unicycle, LTI
from safety import MMSafety

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
    # goals = np.random.uniform(-5, 5, (2, 3))
    goal = np.array([[-4, 0, 3, 0]]).T
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

def test_safety():
    # for computing cartesian position difference
    Ch = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0]]) # mapping human state to [x, y]
    Cr = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]]) # mapping robot state to [x, y]
    
    # NOTE: need W to be invertible for safety controller to work
    W = np.diag([0.3, 0.1, 0.3, 0.1])
    # W = np.diag([0, 0, 0, 0])
    h_dyn = LTI(0.1, W=W)
    r_dyn = Unicycle(0.1)
    safe_controller = MMSafety(r_dyn, h_dyn)

    # randomly initialize 3 goals
    goals = np.random.uniform(-10, 10, (2, 3))
    h_goal = goals[:,[0]]
    r_goal = goals[:,1]

    # initial positions
    xh0 = np.array([[0, 0, 0, 0]]).T
    xr0 = np.array([[-0.5, 0, 0, 0]]).T

    T = 10 # in seconds
    N = int(T / h_dyn.ts)

    # initialize belief
    belief = np.ones(goals.shape[1]) / goals.shape[1]
    sigmas = [W.copy() for _ in range(goals.shape[1])]

    fig, ax = plt.subplots()
    xh_traj = xh0
    xr_traj = xr0
    # simulate for T seconds
    for i in range(N):
        uh = h_dyn.compute_control(xh0, Ch.T @ h_goal, Cr @ xr0)
        ur_ref = r_dyn.compute_goal_control(xr0, r_goal)

        # compute safe control
        ur_safe = safe_controller(xr0, xh0, ur_ref, goals, belief, sigmas)

        # step dynamics forward
        xh0 = h_dyn.step(xh0, uh)
        xr0 = r_dyn.step(xr0, ur_safe)

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))

        # plot
        ax.cla()
        overlay_timesteps(ax, xh_traj, xr_traj, n_steps=i+1)
        ax.scatter(xh0[0], xh0[2], c="blue")
        ax.scatter(xr0[0], xr0[1], c="red")
        ax.scatter(goals[0], goals[1], c="green")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)

if __name__ == "__main__":
    # test_unicycle_goal_reach()
    # test_lti()
    # np.random.seed(0)
    test_safety()
