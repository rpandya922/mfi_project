import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import cvxpy as cp
from scipy.linalg import sqrtm
import pickle
from tqdm import tqdm

from dynamics import Unicycle, LTI
from safety import MMSafety, MMLongTermSafety, SEASafety, BaselineSafety, BaselineSafetySamples
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

def compute_bounding_ellipse(h_dyn, xh0, xr0, Ch, Cr, sigmas, goals):
    # compute smallest ellipse that contains all 3 ellipses defined by sigmas
    Ai_s = []
    bi_s = []
    ci_s = []
    for goal_idx in range(goals.shape[1]):
        goal = goals[:,[goal_idx]]
        Ai = np.linalg.inv(sigmas[goal_idx][[0,2]][:,[0,2]])
        uh_goal = h_dyn.compute_control(xh0, Ch.T @ goal, Cr @ xr0)
        xh_next = h_dyn.step_mean(xh0, uh_goal)
        xh_next = xh_next[[0,2]]
        bi = -Ai.T @ xh_next
        ci = xh_next.T @ Ai @ xh_next - 1

        Ai_s.append(Ai)
        bi_s.append(bi)
        ci_s.append(ci.flatten())
    # solution from Boyd "Convex Optimization" 8.4.1
    n = 2
    Asq = cp.Variable((n,n), symmetric=True)
    btilde = cp.Variable((n,1))
    tau = cp.Variable(len(Ai_s))
    objective = cp.Maximize(cp.log_det(Asq)) # minimize log det A^-1 = -log det A = maximize log det A
    constraints = []
    for i in range(len(Ai_s)):
        c1 = cp.hstack([Asq - tau[i]*Ai_s[i], btilde - tau[i]*bi_s[i], np.zeros((n,n))])
        c2 = cp.hstack([(btilde-tau[i]*bi_s[i]).T, -np.ones((1,1))-tau[i]*ci_s[i], btilde.T])
        c3 = cp.hstack([np.zeros((n,n)), btilde, -Asq])
        constr = (cp.vstack([c1, c2, c3]) << 0)
        constraints.append(constr)

    constraints.append(Asq >> 0)
    constraints.append(tau >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    # print(prob.status)
    # convert Asq and btilde to A and b
    A = sqrtm(Asq.value)
    b = np.linalg.inv(A) @ btilde.value
    # convert A,b into ellipse (i.e. into center and covariance matrix)
    sigma_enclose = np.linalg.inv(A.T @ A)
    c_enclose = -np.linalg.inv(A) @ b

    return c_enclose, sigma_enclose

def run_trajectory(controller : str = "multimodal", change_h_goal = True, plot=True, n_goals=5):
# def visualize_uncertainty():
    # for computing cartesian position difference
    Ch = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0]]) # mapping human state to [x, y]
    Cr = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]]) # mapping robot state to [x, y]
    
    # NOTE: need W to be invertible for safety controller to work
    W = np.diag([0.5, 0.01, 0.5, 0.01])
    # W = np.diag([0, 0, 0, 0])
    h_dyn = LTI(0.1, W=W)
    r_dyn = Unicycle(0.1, kv=2, kpsi=1.2)
    dmin = 1
    if controller == "baseline":
        safe_controller = BaselineSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    elif controller == "multimodal":
        safe_controller = MMSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    elif controller == "SEA":
        safe_controller = SEASafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)

    # safe_controller = MMLongTermSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    # safe_controller = BaselineSafetySamples(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    use_ell_bound = False # whether to compute the bounding ellipse
    phis = []
    safety_actives = []
    distances = []
    all_slacks = np.zeros((0, n_goals))
    h_goal_dists = []
    r_goal_dists = []
    h_goal_reached = []
    r_goal_reached = []
    u_refs = []
    u_safes = []
    all_Ls = []
    all_Ss = []

    # randomly initialize 3 goals
    goals = np.random.uniform(-10, 10, (2, n_goals))
    h_goal_idx = np.random.randint(0, n_goals)
    h_goal = goals[:,[h_goal_idx]]
    r_goal_idx = np.random.randint(0, n_goals)
    r_goal = goals[:,r_goal_idx]

    # initial positions
    # xh0 = np.array([[0, 0, 0, 0]]).T
    # xr0 = np.array([[-0.5, 0, 0, 0]]).T
    # initial positions
    xh0 = np.random.uniform(-10, 10, (4,1))
    xh0[[1,3]] = np.zeros((2,1))
    xr0 = np.random.uniform(-10, 10, (4,1))
    xr0[[2,3]] = np.zeros((2,1))
    # xr0 = np.array([[-5, 0, -5, 0]]).T
    distances.append(np.linalg.norm((Cr@xr0) - (Ch@xh0)))

    T = 25 # in seconds
    N = int(T / h_dyn.ts)

    # robot's belief about the human's goal
    prior = np.ones(goals.shape[1]) / goals.shape[1]
    belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=0.0005)
    # belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=1e-6)
    beliefs = prior
    r_sigma = np.diag([0.7, 0.01, 0.3, 0.01])
    sigmas_init = [r_sigma.copy() for _ in range(goals.shape[1])]

    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
        axes = np.array(axes).flatten()
        ax = axes[0]
        # make ax equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        belief_ax = axes[1]
        phi_ax = axes[2]
        dist_ax = axes[3]
        slack_ax = axes[4]
        control_ax = axes[5]

    xh_traj = xh0
    xr_traj = xr0
    # simulate for T seconds
    for idx in range(N):
        uh = h_dyn.compute_control(xh0, Ch.T @ h_goal, Cr @ xr0)
        ur_ref = r_dyn.compute_goal_control(xr0, r_goal)
        # ur_ref = np.zeros((2,1))

        # compute covariance matrix that's in the direction of the goal
        sigmas = []
        for goal_idx in range(goals.shape[1]):
            # compute angle between human and goal
            goal = goals[:,[goal_idx]]
            # compute human's next state wrt this goal
            uh_goal = h_dyn.compute_control(xh0, Ch.T @ goal, Cr @ xr0)
            xh_next = h_dyn.step_mean(xh0, uh_goal)
            angle = np.arctan((goal[1,0] - xh_next[2,0])/(goal[0,0] - xh_next[0,0]))
            # angle = np.arctan((goal[1,0] - xh0[2,0])/(goal[0,0] - xh0[0,0]))
            # compute covariance matrix from sigmas[goal_idx] rotated by angle
            sigma = sigmas_init[goal_idx][[0,2]][:,[0,2]]
            R = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
            sigma_rot = R @ sigma @ R.T
            # set x & y components of sigmas[goal_idx] to sigmas_rot
            sigma_new = sigmas_init[goal_idx].copy()
            sigma_new[0,0] = sigma_rot[0,0]
            sigma_new[0,2] = sigma_rot[0,1]
            sigma_new[2,0] = sigma_rot[1,0]
            sigma_new[2,2] = sigma_rot[1,1]
            sigmas.append(sigma_new)

        if use_ell_bound:
            if idx == 0:
                c_enclose, sigma_enclose = compute_bounding_ellipse(h_dyn, xh0, xr0, Ch, Cr, sigmas, goals)
            else:
                # compute average next state
                xh_nexts = []
                for goal_idx in range(goals.shape[1]):
                    goal = goals[:,[goal_idx]]
                    uh_goal = h_dyn.compute_control(xh0, Ch.T @ goal, Cr @ xr0)
                    xh_next = h_dyn.step_mean(xh0, uh_goal)
                    xh_nexts.append(xh_next[[0,2]])
                xh_nexts = np.array(xh_nexts)
                c_enclose = np.mean(xh_nexts, axis=0)

        # compute safe control
        if type(safe_controller) == MMLongTermSafety:
            ur_ref = lambda xr, xh: r_dyn.compute_goal_control(xr, r_goal)
        if plot:
            ax.cla()
        else:
            control_ax = None
        # TODO: save reference control, safe control, and control constraints
        ur_safe, phi, safety_active, slacks, Ls, Ss = safe_controller(xr0, xh0, ur_ref, goals, belief.belief, sigmas, return_slacks=True, time=idx, ax=None, return_constraints=True)
        if slacks is None:
            slacks = np.zeros(n_goals)
        # ur_safe, phi, safety_active, slacks = ur_ref, 0, False, np.zeros(3)

        # update robot's belief
        # uh_d = -h_dyn.Kd @ (xh0 - Ch.T @ h_goal)
        belief.update_belief(xh0, uh)
        # belief.belief = np.ones(3) / 3
        # belief.belief = np.array([1, 0, 0])

        # step dynamics forward
        xh0 = h_dyn.step(xh0, uh)
        xr0 = r_dyn.step(xr0, ur_safe)
        # xr0 = r_dyn.step(xr0, ur_ref)

        # change human's goal if applicable
        goal_dist = np.linalg.norm(xh0[[0,2]] - h_goal)
        if change_h_goal and goal_dist < 0.3:
            h_goal_reached.append(h_goal_idx)
            # create a new goal
            goals[:,[h_goal_idx]] = np.random.uniform(-10, 10, (2,1))
            h_goal_idx = (h_goal_idx + 1) % goals.shape[1]
            h_goal = goals[:,[h_goal_idx]]
        else:
            h_goal_reached.append(-1)

        # change robot's goal if applicable
        goal_dist = np.linalg.norm(xr0[[0,1]] - r_goal[:,None])
        if goal_dist < 0.3:
            r_goal_reached.append(r_goal_idx)
            goals[:,[r_goal_idx]] = np.random.uniform(-10, 10, (2, 1))
            # r_goal_idx = (r_goal_idx + 1) % goals.shape[1]
            # r_goal_idx = np.random.randint(0, goals.shape[1])
            # r_goal = goals[:,r_goal_idx]
        else:
            r_goal_reached.append(-1)

        
        # h_dists = np.linalg.norm(xh0[[0,2]] - goals, axis=0)
        # h_goal_idx = np.argmin(h_dists)
        h_goal = goals[:,[h_goal_idx]] # necessary for changing goals

        # robot's goal is closest to itself
        r_dists = np.linalg.norm(xr0[[0,1]] - goals, axis=0)
        r_goal_idx = np.argmin(r_dists)
        r_goal = goals[:,r_goal_idx]

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        xr_traj = np.hstack((xr_traj, xr0))
        beliefs = np.vstack((beliefs, belief.belief))
        phis.append(phi)
        safety_actives.append(safety_active)
        distances.append(np.linalg.norm(Cr@xr0 - Ch@xh0))
        all_slacks = np.vstack((all_slacks, slacks))
        h_goal_dists.append(np.linalg.norm(xh0[[0,2]] - h_goal))
        r_goal_dists.append(np.linalg.norm(xr0[[0,1]] - r_goal[:,None]))
        u_refs.append(ur_ref)
        u_safes.append(ur_safe)
        all_Ls.append(Ls)
        all_Ss.append(Ss)

        if plot:
            # plot
            goal_colors = ["#3A637B", "#C4A46B", "#FF5A00", "#a3b18a"]
            slack_ax.cla()
            for i in range(n_goals):
                slack_ax.plot(3 - all_slacks[:,i], label=f"(k-s)-sigma (g{i})", c=goal_colors[i])
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
            for i in range(n_goals):
                belief_ax.plot(beliefs[:,i], label=f"P(g{i})", c=goal_colors[i])
            belief_ax.legend()

            # ax.cla()
            # compute ellipses for each goal
            k_sigmas = 3 - slacks
            for i, sigma in enumerate(sigmas):
                k = k_sigmas[i]
                # compute k-sigma ellipse
                sigma = sigma[[0,2]][:,[0,2]]
                eigenvalues, eigenvectors = np.linalg.eig(sigma)
                sqrt_eig = np.sqrt(eigenvalues)
                # use only xy components
                # sqrt_eig = sqrt_eig[[0,2]]
                # eigenvectors = eigenvectors[:,[0,2]]
                # compute angle of ellipse
                theta = np.arctan(eigenvectors[1,0]/eigenvectors[0,0])
                # compute human's next state wrt this goal
                uh_i = h_dyn.compute_control(xh0, Ch.T @ goals[:,[i]], Cr @ xr0)
                xh_next = h_dyn.step_mean(xh0, uh_i)
                # compute ellipse
                ellipse = Ellipse(xy=Ch@xh_next, width=2*k*sqrt_eig[0], height=2*k*sqrt_eig[1], angle=np.rad2deg(theta), alpha=0.2, color=goal_colors[i])
                ax.add_patch(ellipse)
            overlay_timesteps(ax, xh_traj, xr_traj, n_steps=idx+1)
            ax.scatter(xh0[0], xh0[2], c="blue")
            heading = xr0[3]
            ax.scatter(xr0[0], xr0[1], c="red", marker=(3, 0, 180*heading/np.pi+30), s=150)
            ax.scatter(goals[0], goals[1], c=goal_colors)
            if use_ell_bound:
                # plot enclosing ellipse in red outline
                eigenvalues, eigenvectors = np.linalg.eig(sigma_enclose)
                sqrt_eig = np.sqrt(eigenvalues)
                theta = np.arctan(eigenvectors[1,0]/eigenvectors[0,0])
                ellipse = Ellipse(xy=c_enclose, width=2*3*sqrt_eig[0], height=2*3*sqrt_eig[1], angle=np.rad2deg(theta), color="red", fill=False)
                ax.add_patch(ellipse)
            # draw dmin circle
            circle = plt.Circle((xh0[0], xh0[2]), dmin, color='k', fill=False, linestyle="--")
            ax.add_artist(circle)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

            control_ax.cla()
            control_ax.plot(r_goal_dists, label="r_goal_dist")
            control_ax.plot(h_goal_dists, label="h_goal_dist")
            control_ax.legend()

            plt.pause(0.001)
            # save figures for video
            # img_path = f"./data/baseline_ctl/"
            # if not os.path.exists(img_path):
            #     os.makedirs(img_path)
            # img_path += f"/{idx:03d}.png"
            # plt.savefig(img_path, dpi=300)

    if plot:
        plt.show()
    
    ret = {"xh_traj": xh_traj, "xr_traj": xr_traj, "phis": phis, "safety_actives": safety_actives, "beliefs": beliefs, "distances": distances, "all_slacks": all_slacks, "h_goal_dists": h_goal_dists, "r_goal_dists": r_goal_dists, "h_goal_reached": h_goal_reached, "r_goal_reached": r_goal_reached, "u_refs": u_refs, "u_safes": u_safes, "all_Ls": all_Ls, "all_Ss": all_Ss}
    return ret

def simulate_all(filepath="./data/sim_stats.pkl"):
    # TODO: re-run with new random goals added for both agents after goals are reached
    n_sim = 2
    controllers = ["baseline", "multimodal", "SEA"]
    all_stats = {controller: [] for controller in controllers}
    for controller in controllers:
        np.random.seed(4)
        controller_stats = []
        for i in tqdm(range(n_sim)):
            res = run_trajectory(controller=controller, plot=False)
            controller_stats.append(res)
        # save stats
        all_stats[controller] = controller_stats
    # save all_stats
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(all_stats, f)

if __name__ == "__main__":
    # np.random.seed(4) # standard seed
    # run_trajectory(controller="multimodal")
    filepath = f"./data/sim_stats_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    simulate_all(filepath)