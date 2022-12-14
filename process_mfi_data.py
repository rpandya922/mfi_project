import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from intention_utils import overlay_timesteps

# nominal goal positions
g1 = np.array([0.32, 0.21])
g2 = np.array([0.59, 0.22])

if __name__ == "__main__":
    user_names = ["p3"]
    modes = [1, 2, 3, 4]
    os.chdir("./kinova/data/pilot_study")
    for user_name in user_names:
        for mode in modes:
            # filepath = "./kinova/data/pilot_study/" + user_name + "/mode" + str(mode)
            filepath = user_name + "/mode" + str(mode)
            frames_path = filepath + "_frames"
            # create directory to save frames if it doesn't exist
            if not os.path.exists(frames_path):
                os.makedirs(frames_path)

            # read in wrist pose data
            df = pd.read_csv(filepath + "_wrist_pose.txt", sep=" ", header=None)
            wrist_pos = df.to_numpy()

            # read in robot pose data
            df = pd.read_csv(filepath + "_robot_pose.txt", sep=" ", header=None)
            rob_pos = df.to_numpy()

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
                # plot goals
                ax.scatter(g1[0], g1[1], c="green", s=75)
                ax.scatter(g2[0], g2[1], c="green", s=75)

                ax.set_aspect('equal', adjustable='box')
                # ax.set_xlim(-0.2, 1.2)
                # ax.set_ylim(-0.5, 0.5)

                img_path = f"{frames_path}/{i:03d}.png"
                plt.savefig(img_path, dpi=200)
                # plt.pause(0.01)

                xh0 = wrist_pos[i,:2]
                xh_traj.append(xh0)
                xr0 = rob_pos[i,:2]
                xr_traj.append(xr0)
            print(f"done with {user_name} mode {mode}")
            # create video with python subprocess with this unix command: ffmpeg -framerate 20 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -vb 20M out.mp4
            # change into frames directory and run the command
            os.chdir(frames_path)
            os.system("ffmpeg -framerate 20 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -vb 20M out.mp4")
            os.chdir("../..")

            # plt.show()