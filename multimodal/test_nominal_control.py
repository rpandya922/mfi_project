import numpy as np
import matplotlib.pyplot as plt

from dynamics import Unicycle
from simulate import overlay_timesteps

if __name__ == "__main__":
    # initial positions
    # xr0 = np.array([[-0.5, 0, 0, np.pi]]).T
    xr0 = np.array([[ 5.58765844],
                    [-6.04629851],
                    [ 0.        ],
                    [ 0.        ]])
    goals = np.array([[ 9.34059678,  0.94464498,  9.4536872 ],
                      [ 4.29631987,  3.95457649, -5.67821009]])
    r_goal_idx = 0
    r_goal = goals[:, r_goal_idx]
    # r_goal = np.array([3, 3])
    r_dyn = Unicycle(0.1, kv=2, kpsi=1.2)

    xr_traj = xr0

    fig, ax = plt.subplots()

    for i in range(200):
        # plot
        ax.cla()
        overlay_timesteps(ax, [], xr_traj)
        heading = xr0[3]
        ax.scatter(xr0[0], xr0[1], c="red", marker=(3, 0, 180*heading/np.pi+30), s=150)
        ax.scatter(r_goal[0], r_goal[1])
        ax.set_xlim([-11, 11])
        ax.set_ylim([-11, 11])
        plt.pause(0.01)

        ur_ref = r_dyn.compute_goal_control(xr0, r_goal)
        
        # step dynamics forward
        xr0 = r_dyn.step(xr0, ur_ref)

        # change robot's goal if applicable
        goal_dist = np.linalg.norm(xr0[[0,1]] - r_goal[:,None])
        if goal_dist < 0.3:
            r_goal_idx = (r_goal_idx + 1) % goals.shape[1]
            r_goal = goals[:,r_goal_idx]

        # save traj
        xr_traj = np.hstack((xr_traj, xr0))
    plt.show()