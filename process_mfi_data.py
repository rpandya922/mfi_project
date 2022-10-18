import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from intention_utils import overlay_timesteps

if __name__ == "__main__":
    # read in wrist pose data
    filename = "./kinova/data/wrist_pose3.txt"
    df = pd.read_csv(filename, sep=" ", header=None)
    wrist_pos = df.to_numpy()

    # read in robot pose data
    filename = "./kinova/data/robot_pose3.txt"
    df = pd.read_csv(filename, sep=" ", header=None)
    rob_pos = df.to_numpy()
    rob_pos[:,1] = -rob_pos[:,1]

    fig, ax = plt.subplots()
    xh0 = wrist_pos[0,:2]
    xh_traj = [xh0]
    xr0 = rob_pos[0,:2]
    xr_traj = [xr0]
    # save frames for animation
    for i in range(wrist_pos.shape[0]):
        ax.cla()
        xh_traj_arr = np.array(xh_traj)
        xh_traj_arr = np.insert(xh_traj_arr.T, 1, np.zeros(len(xh_traj)), 0)
        xh_traj_arr = np.insert(xh_traj_arr, 3, np.zeros(len(xh_traj)), 0)
        xr_traj_arr = np.array(xr_traj)
        xr_traj_arr = np.insert(xr_traj_arr.T, 1, np.zeros(len(xr_traj)), 0)
        xr_traj_arr = np.insert(xr_traj_arr, 3, np.zeros(len(xr_traj)), 0)
        overlay_timesteps(ax, xh_traj_arr, xr_traj_arr, [], n_steps=i, h_cmap="Blues", r_cmap="Reds", linewidth=4)
        ax.scatter(xh0[0], xh0[1], c="blue", s=75)
        ax.scatter(xr0[0], xr0[1], c="red", s=75)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.5, 0.5)

        # img_path = f"./kinova/data/frames_2d/{i:03d}.png"
        # plt.savefig(img_path, dpi=200)
        plt.pause(0.01)

        xh0 = wrist_pos[i,:2]
        xh_traj.append(xh0)
        xr0 = rob_pos[i,:2]
        xr_traj.append(xr0)
    print("done")
    plt.show()