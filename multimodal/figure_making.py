import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle

from dynamics import LTI, Unicycle
from bayes_inf import BayesEstimator
from safety import MMSafety, BaselineSafety, SEASafety
from simulate import overlay_timesteps

def plot_constraints(controller, xr0, xh0, ur_ref, goals, belief, sigmas):

    ur_safe, phi, safety_active, slacks, Ls, Ss = controller(xr0, xh0, ur_ref, goals, belief.belief, sigmas, return_slacks=True, ax=None, return_constraints=True)
    Ls = np.array(Ls)
    Ss = np.array(Ss)

    fig, ax = plt.subplots()
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_aspect("equal")

    halfspaces = np.hstack((Ls, -Ss[:,None]))

    x = np.linspace(-30, 30, 100)
    symbols = ['-', '+', 'x', '*']
    signs = [-1, -1, -1]
    fmt = {"color": "#8a7fad", "edgecolor": "b", "alpha": 0.3}
    for h, sym, sign in zip(halfspaces, symbols, signs):
        hlist = h.tolist()
        # fmt["hatch"] = sym
        if h[1]== 0:
            ax.axvline(-h[2]/h[0], label='{}x+{}y+{}=0'.format(*hlist))
            xi = np.linspace(xlim[sign], -h[2]/h[0], 100)
            ax.fill_between(xi, ylim[0], ylim[1], **fmt)
        else:
            ax.plot(x, (-h[2]-h[0]*x)/h[1], label='{}x+{}y+{}=0'.format(*hlist))
            ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], **fmt)

    plt.scatter(ur_ref[0,0], ur_ref[1,0], marker='x', color='r', label='ur_ref', s=200)
    plt.scatter(ur_safe[0,0], ur_safe[1,0], marker='x', color='g', label='ur_safe', s=200)

    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)

def plot_control_space():
    np.random.seed(0)
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
    baseline_controller = BaselineSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    mm_controller = MMSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    sea_controller = SEASafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)

    # randomly initialize 3 goals
    goals = np.random.uniform(-10, 10, (2, 3))
    h_goal_idx = 0
    h_goal = goals[:,[h_goal_idx]]
    r_goal_idx = 0
    r_goal = goals[:,r_goal_idx]

    # initial positions
    xh0 = np.random.uniform(-10, 10, (4,1))
    xh0[[1,3]] = np.zeros((2,1))
    xr0 = np.random.uniform(-10, 10, (4,1))
    xr0[[2,3]] = np.zeros((2,1))

    T = 25 # in seconds
    N = int(T / h_dyn.ts)

    # robot's belief about the human's goal
    prior = np.ones(goals.shape[1]) / goals.shape[1]
    belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=0.0005)
    # belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=1e-6)
    beliefs = prior
    r_sigma = np.diag([0.7, 0.01, 0.3, 0.01])
    sigmas_init = [r_sigma.copy() for _ in range(goals.shape[1])]

    xh_traj = xh0
    xr_traj = xr0
    
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

        # TODO: save reference control, safe control, and control constraints
        ur_safe, phi, safety_active, slacks, Ls, Ss = baseline_controller(xr0, xh0, ur_ref, goals, belief.belief, sigmas, return_slacks=True, time=idx, ax=None, return_constraints=True)

        if safety_active:
            # plot control constraints for all controllers
            plot_constraints(baseline_controller, xr0, xh0, ur_ref, goals, belief, sigmas)
            plot_constraints(mm_controller, xr0, xh0, ur_ref, goals, belief, sigmas)
            plot_constraints(sea_controller, xr0, xh0, ur_ref, goals, belief, sigmas)
            plt.show()
            1/0
    
        # update robot's belief
        belief.update_belief(xh0, uh)

        # step dynamics forward
        xh0 = h_dyn.step(xh0, uh)
        xr0 = r_dyn.step(xr0, ur_safe)

def plot_bayes_inf():
    np.random.seed(5)
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
    baseline_controller = BaselineSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    mm_controller = MMSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    sea_controller = SEASafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)

    # randomly initialize 3 goals
    goals = np.random.uniform(-10, 10, (2, 3))
    h_goal_idx = 0
    h_goal = goals[:,[h_goal_idx]]
    r_goal_idx = 0
    r_goal = goals[:,r_goal_idx]

    # initial positions
    xh0 = np.random.uniform(-10, 10, (4,1))
    xh0[[1,3]] = np.zeros((2,1))
    xr0 = np.random.uniform(-10, 10, (4,1))
    xr0[[2,3]] = np.zeros((2,1))

    T = 25 # in seconds
    N = int(T / h_dyn.ts)

    # robot's belief about the human's goal
    prior = np.ones(goals.shape[1]) / goals.shape[1]
    # belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=0.0005)
    belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=0.001)
    # belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=1e-6)
    beliefs = prior
    r_sigma = np.diag([0.7, 0.01, 0.3, 0.01])
    sigmas_init = [r_sigma.copy() for _ in range(goals.shape[1])]
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

    xh_traj = xh0
    
    for idx in range(N):
        uh = h_dyn.compute_control(xh0, Ch.T @ h_goal, Cr @ xr0)

        # update robot's belief
        belief.update_belief(xh0, uh)

        # step dynamics forward
        xh0 = h_dyn.step(xh0, uh)

        # save data
        xh_traj = np.hstack((xh_traj, xh0))
        beliefs = np.vstack((beliefs, belief.belief))

        if np.linalg.norm(xh0[[0,2]] - goals[:,[h_goal_idx]]) < 0.3:
            break

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes = axes.flatten()
    axes[0].set_xlim(-10, 10)
    axes[0].set_ylim(-10, 10)
    axes[0].set_aspect('equal', adjustable='box')
    overlay_timesteps(axes[0], xh_traj, [], n_steps=idx+1)
    axes[0].scatter(goals[0,:], goals[1,:], color=goal_colors, label='goals', marker='x', s=200)
    axes[0].scatter(xh_traj[0,-1], xh_traj[2,-1], color='b', s=200)

    axes[1].plot(beliefs[:,0], c=goal_colors[0])
    axes[1].plot(beliefs[:,1], c=goal_colors[1])
    axes[1].plot(beliefs[:,2], c=goal_colors[2])

    plt.show()

def plot_front_figure():
    # for computing cartesian position difference
    Ch = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0]]) # mapping human state to [x, y]
    Cr = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]]) # mapping robot state to [x, y]
    
    # NOTE: need W to be invertible for safety controller to work
    W = np.diag([0.5, 0.01, 0.5, 0.01])
    # W = np.diag([0, 0, 0, 0])
    h_dyn = LTI(0.2, W=W, gamma=0)
    r_dyn = Unicycle(0.2, kv=2, kpsi=1.2)
    dmin = 1
    baseline_controller = BaselineSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)
    mm_controller = MMSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=10)
    sea_controller = SEASafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)

    # randomly initialize 3 goals
    n_goals = 3
    goals = np.array([[-7, 8], [5,-7], [5, 5]]).T
    h_goal_idx = 1
    h_goal = goals[:,[h_goal_idx]]
    # r_goal_idx = 2
    # r_goal = goals[:,r_goal_idx]
    r_goal = np.array([-11, -11])

    # initial positions
    xh0 = np.array([[-7, 1, -7, 0]]).T
    # xr0 = np.array([[-4, -4, 0.5, 3*np.pi/2]]).T
    xr0 = np.array([[-4.5, -4.5, 0.5, 3*np.pi/2]]).T

    T = 25 # in seconds
    N = int(T / h_dyn.ts)

    # robot's belief about the human's goal
    # prior = np.array([1.0, 0.01, 0.01])
    prior = np.array([0.5, 1.0, 0.01])
    # prior = np.array([0.01, 0.01, 1.0])
    prior = prior / np.sum(prior)
    # belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=0.0005)
    belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=0.001)
    # belief = BayesEstimator(Ch.T @ goals, h_dyn, prior=prior, beta=1e-6)
    beliefs = prior
    r_sigma0 = np.diag([0.7, 0.01, 0.1, 0.01])
    r_sigma1 = np.diag([0.7, 0.01, 0.1, 0.01])
    r_sigma2 = np.diag([0.7, 0.01, 0.1, 0.01])
    sigmas_init = [r_sigma0, r_sigma1, r_sigma2]
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

    xh_traj = xh0

    fig, ax = plt.subplots()

    uh = h_dyn.compute_control(xh0, Ch.T @ h_goal, Cr @ xr0)
    ur_ref = r_dyn.compute_goal_control(xr0, r_goal)

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

    ur_safe, phi, safety_active, slacks, Ls, Ss = mm_controller(xr0, xh0, ur_ref, goals, belief.belief, sigmas, return_slacks=True, time=0, ax=None, return_constraints=True)
    print(ur_ref, ur_safe)

    k_sigmas = 3 - slacks
    print(k_sigmas)
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
        ellipse = Ellipse(xy=Ch@xh_next, width=2*k*sqrt_eig[0], height=2*k*sqrt_eig[1], angle=np.rad2deg(theta), alpha=0.3, color=goal_colors[i])
        ax.add_patch(ellipse)

    k_sigmas = np.ones_like(slacks)*3
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
        ellipse = Ellipse(xy=Ch@xh_next, width=2*k*sqrt_eig[0], height=2*k*sqrt_eig[1], angle=np.rad2deg(theta), alpha=0.7, color=goal_colors[i], fill=False, linestyle='--')
        ax.add_patch(ellipse)

    ax.scatter(xh0[0], xh0[2], c="blue")
    heading = xr0[3]
    # ax.scatter(xr0[0], xr0[1], c="red", marker=(3, 0, 180*heading/np.pi+30), s=150)
    ax.scatter(goals[0], goals[1], c=goal_colors, marker="x", s=200)
    ax.scatter([r_goal[0]], [r_goal[1]], c="k", marker="x", s=200)

    # draw dmin circle
    circle = plt.Circle((xh0[0], xh0[2]), dmin, color='k', fill=False, linestyle="--")
    ax.add_artist(circle)

    # sample controls
    # us = np.array([-100, -50, -10, 0, 10, 50, 100])
    u1s = np.linspace(-100, 100, 100)
    u2s = np.linspace(-2*np.pi, 2*np.pi, 100)
    U1, U2 = np.meshgrid(u1s, u2s)
    U = np.vstack((U1.flatten(), U2.flatten()))
    # n_actions = 16
    # angles = np.linspace(0, 2 * np.pi, num=(n_actions + 1))[:-1]
    # U = []
    # for r in [50, 70]:
    #     actions = np.array([r * np.cos(angles), r * np.sin(angles)]).T
    #     U.append(actions)
    # U = np.vstack(U).T
    # compute whether these controls are safe
    S_min = np.amin(Ss)
    fun = lambda u: S_min - (Ls[0] @ u)
    is_safe = fun(U) >= 0

    # compute distance from each control to the halfspace constraint (to check how safe it is)
    a = Ls[0][0]
    b = Ls[0][1]
    c = -S_min
    dists = np.abs(a*U[0] + b*U[1] + c) / np.sqrt(a**2 + b**2)
    
    # compute next state given each potential control
    xr_nexts = []
    for u in U.T:
        xr_nexts.append(r_dyn.step(xr0, u[:,None]))
    xr_nexts = np.array(xr_nexts).squeeze().T

    # # plot unsafe next states in red
    # cmap = matplotlib.cm.get_cmap("Reds")
    # norm = matplotlib.colors.Normalize(vmin=-1, vmax=dists[~is_safe].max())
    # for i, u in enumerate(U.T):
    #     if is_safe[i]:
    #         continue
    #     # ax.plot([xr0[0,0], xr_nexts[0,i]], [xr0[1,0], xr_nexts[1,i]], color=cmap(norm(dists[i])), alpha=0.1)
    #     heading = xr_nexts[3,i]
    #     ax.scatter(xr_nexts[0,i], xr_nexts[1,i], color=cmap(norm(dists[i])), marker=(3, 0, 180*heading/np.pi+30), alpha=0.1, s=100)
    #     # ax.scatter(xr_nexts[0,i], xr_nexts[1,i], color=cmap(norm(dists[i])), alpha=0.1, s=100, edgecolors='none')

    # # plot safe next states in green
    # cmap = matplotlib.cm.get_cmap("Greens")
    # norm = matplotlib.colors.Normalize(vmin=-5, vmax=dists[is_safe].max())
    # for i, u in enumerate(U.T):
    #     if not is_safe[i]:
    #         continue
    #     # ax.plot([xr0[0,0], xr_nexts[0,i]], [xr0[1,0], xr_nexts[1,i]], color=cmap(norm(dists[i])), alpha=0.1)
    #     heading = xr_nexts[3,i]
    #     ax.scatter(xr_nexts[0,i], xr_nexts[1,i], color=cmap(norm(dists[i])), marker=(3, 0, 180*heading/np.pi+30), alpha=0.1, s=100)
    #     # ax.scatter(xr_nexts[0,i], xr_nexts[1,i], color=cmap(norm(dists[i])), alpha=1, s=100, edgecolors='none')

    unsafe_nexts = xr_nexts[:,~is_safe]
    ax.scatter(unsafe_nexts[0], unsafe_nexts[1], c=dists[~is_safe], cmap="Reds", vmin=-5, vmax=dists[~is_safe].max(), alpha=0.05, s=100, edgecolors='none')

    safe_nexts = xr_nexts[:,is_safe]
    ax.scatter(safe_nexts[0], safe_nexts[1], c=dists[is_safe], cmap="Greens", vmin=-5, vmax=dists[is_safe].max(), alpha=0.05, s=100, edgecolors='none')
    
    heading = xr0[3]
    ax.scatter(xr0[0], xr0[1], c="darkred", marker=(3, 0, 180*heading/np.pi+30), s=100)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    plt.show()

def plot_traj():
    filename = "./data/traj_16.pkl"
    with open(filename, "rb") as f:
        all_data = pickle.load(f)
    controller = "SEA"
    data = all_data[controller]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 2.2))
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes = axes.flatten()
    ax = axes[0]
    belief_ax = axes[1]
    xh_traj = np.array(data["xh_traj"])
    xr_traj = np.array(data["xr_traj"])
    dists = np.array(data["distances"])
    all_slacks = np.array(data["all_slacks"])

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
    mm_controller = MMSafety(r_dyn, h_dyn, dmin=dmin, eta=0.5, k_phi=5)

    # goals = np.array(data["goals"])
    goals = np.array([[1.5, -8.5], [-9, -6], [5.5, -5]]).T
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]
    beliefs = np.array(data["beliefs"])
    r_sigma = np.diag([0.7, 0.01, 0.3, 0.01])
    sigmas_init = [r_sigma.copy() for _ in range(goals.shape[1])]

    viol_idx = 58 # timestep that safety was violated
    xh0 = xh_traj[:,[viol_idx]]
    xr0 = xr_traj[:,[viol_idx]]

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

    overlay_timesteps(ax, xh_traj[:,:viol_idx+1], xr_traj[:,:viol_idx+1], n_steps=viol_idx)
    ax.scatter(xh0[0], xh0[2], c="blue")
    ax.scatter(xr0[0], xr0[1], c="red")
    ax.scatter(goals[0], goals[1], c=goal_colors, marker="x")

    slacks = all_slacks[viol_idx]

    if controller == "SEA":
        sigmas_loop = [sigmas[np.argmax(beliefs[viol_idx])]]
        slacks = all_slacks[viol_idx]
        alphas = [0.2, 0.2, 0.2]
    else:
        sigmas_loop = sigmas
        ur_ref = data["u_refs"][viol_idx]
        slacks = all_slacks[viol_idx]
        # _, _, _, slacks, _, _ = mm_controller(xr0, xh0, ur_ref, goals, beliefs[viol_idx], sigmas_loop, return_slacks=True, ax=None, return_constraints=True)
        # print(all_slacks[viol_idx])
        # print(slacks)
        # alphas = [0.1, 0.6, 0.1]
        alphas = [0.2, 0.2, 0.2]
    
    k_sigmas = 3 - slacks
    # draw k-sigma ellipse for each
    for i, sigma in enumerate(sigmas_loop):
        sigma = sigmas[i]
        k = k_sigmas[i]
        # compute k-sigma ellipse
        sigma = sigma[[0,2]][:,[0,2]]
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        sqrt_eig = np.sqrt(eigenvalues)
        # compute angle of ellipse
        theta = np.arctan(eigenvectors[1,0]/eigenvectors[0,0])
        # compute human's next state wrt this goal
        uh_i = h_dyn.compute_control(xh0, Ch.T @ goals[:,[i]], Cr @ xr0)
        xh_next = h_dyn.step_mean(xh0, uh_i)
        # compute ellipse
        ellipse = Ellipse(xy=Ch@xh_next, width=2*k*sqrt_eig[0], height=2*k*sqrt_eig[1], angle=np.rad2deg(theta), alpha=alphas[i], color=goal_colors[i])
        ax.add_patch(ellipse)

    # draw dmin circle
    circle = plt.Circle((xh_traj[0,viol_idx], xh_traj[2,viol_idx]), dmin, color="black", fill=False, linestyle="--")
    ax.add_artist(circle)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, -2.5)
    # ax.set_ylim(-10, 10)

    for i in range(goals.shape[1]):
        belief_ax.plot(beliefs[:,i][:viol_idx], c=goal_colors[i][:viol_idx], label=f"P(g{i})")
    belief_ax.legend()

    plt.show()

    # import ipdb; ipdb.set_trace()

    # for i in range(xh_traj.shape[1]):
    #     ax.clear()
    #     ax.scatter(xh_traj[0,i], xh_traj[2,i], label="human")
    #     ax.scatter(xr_traj[0,i], xr_traj[1,i], label="robot")
    #     ax.set_xlim(-10, 10)
    #     ax.set_ylim(-10, 10)
    #     plt.pause(0.001)
    # plt.show()

if __name__ == "__main__":
    plot_control_space()
    # plot_bayes_inf()
    # plot_front_figure()
    # plot_traj()