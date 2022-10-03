import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import ipdb

from intention_utils import overlay_timesteps

if __name__ == "__main__":
    # read in wrist pose data
    filename = "./data/ford_data/3_3/3_3_wrist_pose.txt"
    df = pd.read_csv(filename, sep=" ", header=None)
    wrist_pos = df.to_numpy()

    # read in object pose data
    filename = "./data/ford_data/3_3/3_3_obj_pose.txt"
    df = pd.read_csv(filename, sep=" ", header=None)
    obj_pos = df.to_numpy()
    obj_pos = obj_pos.reshape((obj_pos.shape[0], obj_pos.shape[1]//3, 3))

    # plot full trajectory 
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot(wrist_pos[:,0], wrist_pos[:,1], wrist_pos[:,2])
    # ax.scatter(obj_pos[0,:,0], obj_pos[0,:,1], obj_pos[0,:,2], c='r')
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.set_aspect('equal', adjustable='box')
    # ax.plot(wrist_pos[:,0], wrist_pos[:,1])
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-0.5, 0.5)
    # plt.show()

    fig, ax = plt.subplots()
    xh0 = wrist_pos[0,:2]
    xh_traj = [xh0]
    # save frames for animation
    for i in range(wrist_pos.shape[0]):
        ax.cla()
        xh_traj_arr = np.array(xh_traj)
        xh_traj_arr = np.insert(xh_traj_arr.T, 1, np.zeros(len(xh_traj)), 0)
        xh_traj_arr = np.insert(xh_traj_arr, 3, np.zeros(len(xh_traj)), 0)
        overlay_timesteps(ax, xh_traj_arr, [], [], n_steps=i, h_cmap="Blues", linewidth=4)
        ax.scatter(xh0[0], xh0[1], c="blue", s=75)
        obj = obj_pos[i,:,:]
        ax.scatter(obj[:,0], obj[:,1], c="red", s=75)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)

        img_path = f"./data/ford_data/3_3/frames/{i:03d}.png"
        plt.savefig(img_path, dpi=200)
        # plt.pause(0.01)

        xh0 = wrist_pos[i,:2]
        xh_traj.append(xh0)
    print("done")
    plt.show()