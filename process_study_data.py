import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import pickle
import os
import glob

from intention_utils import overlay_timesteps

def load_user_data(user_id):
    # find folder with this user's id
    foldername = f"./data/self_study/user_{user_id}_*"
    foldername = glob.glob(foldername)[0]
    # load data
    robots = ["baseline", "baseline_belief", "cbp"]
    n_games = 4
    all_data = {robot: [] for robot in robots}
    for robot in robots:
        for game_idx in range(n_games):
            filepath = os.path.join(foldername, f"{robot}_trial{game_idx}.pkl")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            all_data[robot].append(data)
    return all_data

def process_data(user_ids):
    all_data = {user_id: load_user_data(user_id) for user_id in user_ids}
    robot_types = ["baseline", "baseline_belief", "cbp"]

    # scores for each robot type
    scores = {robot_type: [] for robot_type in robot_types}
    # time taken to reach first goal for each robot type
    times = {robot_type: [] for robot_type in robot_types}

    df = pd.DataFrame({"user_id": pd.Series(dtype="int"), "robot_type": pd.Series(dtype="str"), "game_idx": pd.Series(dtype="int"), "team_score": pd.Series(dtype="float")})
    # add to dataframe
    for user_id in user_ids:
        for robot_type in robot_types:
            for game_idx in range(len(all_data[user_id][robot_type])):
                data = all_data[user_id][robot_type][game_idx]
                df = pd.concat([df, pd.DataFrame({"user_id": user_id, "robot_type": robot_type, "game_idx": game_idx, "team_score": data["team_score"]}, index=[0])], ignore_index=True)
    
    # compute repeated measures anova for scores
    aov = pg.rm_anova(dv="team_score", within=["robot_type"], subject="user_id", data=df)
    print(aov)

    # compute post-hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv="team_score", within="robot_type", subject="user_id", padjust="bonf", effsize="cohen")

    # TODO: find any pairs where the difference is significant with corrected p-value
    print(post_hoc)

def plot_traj(user_id):
    user_data = load_user_data(user_id)
    robot_types = ["baseline", "baseline_belief", "cbp"]
    robot_colors = {"baseline": "brown", "baseline_belief": "purple", "cbp": "green"}
    robot_cmaps = {"baseline": "copper_r", "baseline_belief": "Purples", "cbp": "Greens"}

    for robot in robot_types:
        for game_idx in range(len(user_data[robot])):
            data = user_data[robot][game_idx]
            # plot human trajectory
            xh_traj = data["xh_traj"]
            print(xh_traj.shape)
            xr_traj = data["xr_traj"]
            goals = data["goals"]
            fig, ax = plt.subplots()
            for i in range(xh_traj.shape[1]):
                # plot traj trail
                ax.cla()
                if i > 50:
                    xh_traj_i = xh_traj[:,i-50:i+1]
                    xr_traj_i = xr_traj[:,i-50:i+1]
                else:
                    xh_traj_i = xh_traj[:,:i+1]
                    xr_traj_i = xr_traj[:,:i+1]
                overlay_timesteps(ax, xh_traj_i, xr_traj_i, n_steps=50, r_cmap=robot_cmaps[robot])
                ax.scatter(xh_traj[0,i], xh_traj[2,i], c="blue")
                ax.scatter(xr_traj[0,i], xr_traj[2,i], c=robot_colors[robot])
                goals_i = goals[:,:,i]
                ax.scatter(goals_i[0], goals_i[2], c="green")
                ax.set_xlim(-11, 11)
                ax.set_ylim(-11, 8.5)
                ax.set_aspect("equal")
                plt.pause(0.01)
            input(": ")

def plot_obj_selection(user_ids):
    all_data = {user_id: load_user_data(user_id) for user_id in user_ids}
    # get only cbp data
    cbp_data = {user_id: all_data[user_id]["cbp"] for user_id in user_ids}
    # TODO: complete

if __name__ == "__main__":
    user_ids = [4525, 6600, 7998]
    # process_data(user_ids)
    plot_traj(user_ids[0])
