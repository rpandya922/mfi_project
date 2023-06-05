import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse

from dynamics import Unicycle, LTI
from safety import MMSafety, MMLongTermSafety, SEASafety 
from bayes_inf import BayesEstimator

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
    dmin = 1
    safe_controller = MMSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    # safe_controller = SEASafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    phis = []
    safety_actives = []
    distances = []

    # randomly initialize 3 goals
    goals = np.random.uniform(-10, 10, (2, 3))
    h_goal = goals[:,[0]]
    r_goal = goals[:,0]

    # initial positions
    xh0 = np.array([[0, 0, 0, 0]]).T
    xr0 = np.array([[-0.5, 0, 0, 0]]).T
    distances.append(np.linalg.norm((Cr@xr0) - (Ch@xh0)))

    T = 10 # in seconds
    N = int(T / h_dyn.ts)

    # robot's belief about the human's goal
    prior = np.ones(goals.shape[1]) / goals.shape[1]
    belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=1)
    beliefs = prior
    sigmas = [W.copy() for _ in range(goals.shape[1])]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    axes = np.array(axes).flatten()
    ax = axes[0]
    belief_ax = axes[1]
    phi_ax = axes[2]
    dist_ax = axes[3]

    xh_traj = xh0
    xr_traj = xr0
    # simulate for T seconds
    for i in range(N):
        uh = h_dyn.compute_control(xh0, Ch.T @ h_goal, Cr @ xr0)
        ur_ref = r_dyn.compute_goal_control(xr0, r_goal)
        # compute safe control
        ur_safe, phi, safety_active = safe_controller(xr0, xh0, ur_ref, goals, belief.belief, sigmas)

        # update robot's belief
        belief.update_belief(xh0, uh)

        # step dynamics forward
        xh0 = h_dyn.step(xh0, uh)
        xr0 = r_dyn.step(xr0, ur_safe)
        # xr0 = r_dyn.step(xr0, ur_ref)

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))
        beliefs = np.vstack((beliefs, belief.belief))
        phis.append(phi)
        safety_actives.append(safety_active)
        distances.append(np.linalg.norm(Cr@xr0 - Ch@xh0))

        # plot
        dist_ax.cla()
        dist_ax.plot(distances, label="distance")
        dist_ax.plot(np.ones_like(distances) * dmin, c="k", linestyle="--", label="dmin")
        dist_ax.legend()

        phi_ax.cla()
        phi_ax.plot(phis, label="phi")
        phi_ax.plot(np.zeros_like(phis), c="k", linestyle="--")
        # plot a vertical red line at each timestep that safety_active is True
        for j, active in enumerate(safety_actives):
            if active:
                phi_ax.axvline(x=j, c="red", linestyle="--", alpha=0.5)
        phi_ax.legend()

        belief_ax.cla()
        belief_ax.plot(beliefs[:,0], label="P(g0)")
        belief_ax.plot(beliefs[:,1], label="P(g1)")
        belief_ax.plot(beliefs[:,2], label="P(g2)")
        belief_ax.legend()

        ax.cla()
        overlay_timesteps(ax, xh_traj, xr_traj, n_steps=i+1)
        ax.scatter(xh0[0], xh0[2], c="blue")
        ax.scatter(xr0[0], xr0[1], c="red")
        ax.scatter(goals[0], goals[1], c="green")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)

def visualize_uncertainty():
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
    dmin = 1
    safe_controller = MMLongTermSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    # safe_controller = MMSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    # safe_controller = SEASafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    phis = []
    safety_actives = []
    distances = []
    all_slacks = np.zeros((0, 3))

    # randomly initialize 3 goals
    goals = np.random.uniform(-10, 10, (2, 3))
    h_goal = goals[:,[0]]
    r_goal = goals[:,0]

    # initial positions
    xh0 = np.array([[0, 0, 0, 0]]).T
    xr0 = np.array([[-0.5, 0, 0, 0]]).T
    # xr0 = np.array([[-5, 0, -5, 0]]).T
    distances.append(np.linalg.norm((Cr@xr0) - (Ch@xh0)))

    T = 10 # in seconds
    N = int(T / h_dyn.ts)

    # robot's belief about the human's goal
    prior = np.ones(goals.shape[1]) / goals.shape[1]
    belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=1)
    beliefs = prior
    sigmas = [W.copy() for _ in range(goals.shape[1])]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    axes = np.array(axes).flatten()
    ax = axes[0]
    # make ax equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    belief_ax = axes[1]
    phi_ax = axes[2]
    dist_ax = axes[3]
    slack_ax = axes[4]

    xh_traj = xh0
    xr_traj = xr0
    # simulate for T seconds
    for idx in range(N):
        uh = h_dyn.compute_control(xh0, Ch.T @ h_goal, Cr @ xr0)
        ur_ref = r_dyn.compute_goal_control(xr0, r_goal)
        # ur_ref = np.zeros((2,1))
        # compute safe control
        if type(safe_controller) == MMLongTermSafety:
            ur_ref = lambda xr, xh: r_dyn.compute_goal_control(xr, r_goal)
        ur_safe, phi, safety_active, slacks = safe_controller(xr0, xh0, ur_ref, goals, belief.belief, sigmas, return_slacks=True, time=idx)

        # update robot's belief
        belief.update_belief(xh0, uh)
        # belief.belief = np.array([1, 0, 0])

        # step dynamics forward
        xh0 = h_dyn.step(xh0, uh)
        xr0 = r_dyn.step(xr0, ur_safe)
        # xr0 = r_dyn.step(xr0, ur_ref)

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))
        beliefs = np.vstack((beliefs, belief.belief))
        phis.append(phi)
        safety_actives.append(safety_active)
        distances.append(np.linalg.norm(Cr@xr0 - Ch@xh0))
        all_slacks = np.vstack((all_slacks, slacks))

        # plot
        goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
        slack_ax.cla()
        slack_ax.plot(3 - all_slacks[:,0], label="(k-s)-sigma (g0)", c=goal_colors[0])
        slack_ax.plot(3 - all_slacks[:,1], label="(k-s)-sigma (g1)", c=goal_colors[1])
        slack_ax.plot(3 - all_slacks[:,2], label="(k-s)-sigma (g2)", c=goal_colors[2])
        slack_ax.legend()

        dist_ax.cla()
        dist_ax.plot(distances, label="distance")
        dist_ax.plot(np.ones_like(distances) * dmin, c="k", linestyle="--", label="dmin")
        dist_ax.legend()

        phi_ax.cla()
        phi_ax.plot(phis, label="phi")
        phi_ax.plot(np.zeros_like(phis), c="k", linestyle="--")
        # plot a vertical red line at each timestep that safety_active is True
        for j, active in enumerate(safety_actives):
            if active:
                phi_ax.axvline(x=j, c="red", linestyle="--", alpha=0.5)
        phi_ax.legend()

        belief_ax.cla()
        belief_ax.plot(beliefs[:,0], label="P(g0)", c=goal_colors[0])
        belief_ax.plot(beliefs[:,1], label="P(g1)", c=goal_colors[1])
        belief_ax.plot(beliefs[:,2], label="P(g2)", c=goal_colors[2])
        belief_ax.legend()

        ax.cla()
        # compute ellipses for each goal
        k_sigmas = 3 - slacks
        for i, sigma in enumerate(sigmas):
            k = k_sigmas[i]
            # compute k-sigma ellipse 
            eigenvalues, eigenvectors = np.linalg.eig(sigma)
            sqrt_eig = np.sqrt(eigenvalues)
            # use only xy components
            sqrt_eig = sqrt_eig[[0,2]]
            eigenvectors = eigenvectors[:,[0,2]]
            # compute angle of ellipse
            theta = np.arctan2(eigenvectors[1,0], eigenvectors[0,0])
            # compute human's next state wrt this goal
            uh_i = h_dyn.compute_control(xh0, Ch.T @ goals[:,[i]], Cr @ xr0)
            xh_next = h_dyn.step_mean(xh0, uh_i)
            # compute ellipse
            ellipse = Ellipse(xy=Ch@xh_next, width=2*k*sqrt_eig[0], height=2*k*sqrt_eig[1], angle=np.rad2deg(theta), alpha=0.2, color=goal_colors[i])
            ax.add_patch(ellipse)
        overlay_timesteps(ax, xh_traj, xr_traj, n_steps=i+1)
        ax.scatter(xh0[0], xh0[2], c="blue")
        ax.scatter(xr0[0], xr0[1], c="red")
        ax.scatter(goals[0], goals[1], c=goal_colors)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        plt.pause(0.01)

        # save figures for video
        # plt.savefig(f"./data/uncertainty/{idx:03d}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # test_unicycle_goal_reach()
    # test_lti()
    np.random.seed(4)
    # test_safety()
    visualize_uncertainty()
