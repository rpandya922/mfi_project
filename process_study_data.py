import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import pickle
import os
import glob
import pyrankvote
from pyrankvote import Candidate, Ballot

from intention_utils import overlay_timesteps

def load_user_data(user_id):
    # find folder with this user's id
    foldername = f"./data/user_study/user_{user_id}_*"
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

def compute_hesitation_time(data):
    xh_traj = data["xh_traj"]
    # find timesteps goals were reached
    times = np.where(np.array(data["h_goal_reached"]) != -1)[0]
    # compute human state differences
    diffs = np.diff(xh_traj, axis=1)
    # compute norm of differences
    norms = np.linalg.norm(diffs, axis=0)

    # compute once per goal selection (not counting 10 timesteps it takes to get goal)
    hes = 0
    for i in range(len(times)):
        if i == 0:
            goal_norms = norms[:times[i]-10]
        else:
            goal_norms = norms[times[i-1]:times[i]-10]

        # find first contiguous sequence of norms < 0.2
        idx = norms[:times[i]-10] < 0.2
        hes_times = np.split(goal_norms, np.where(np.diff(idx))[0])[0]
        hes += len(hes_times) * 0.1
    return hes / len(times)

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
                hesitation_time = compute_hesitation_time(data)
                df = pd.concat([df, pd.DataFrame({"user_id": user_id, "robot_type": robot_type, "game_idx": game_idx, "team_score": data["team_score"], "hesitation_time": hesitation_time}, index=[0])], ignore_index=True)
                if int(data["n_collisions"]) > 0:
                    print(f"{user_id}, {robot_type}, {game_idx}, {data['n_collisions']}")
    
    # compute repeated measures anova for scores
    aov = pg.rm_anova(dv="team_score", within=["robot_type"], subject="user_id", data=df, correction=True)
    print("Mean scores")
    for robot_type in robot_types:
        print(f"{robot_type}: {df[df.robot_type == robot_type].team_score.mean()}, {df[df.robot_type == robot_type].team_score.std()}")
    print(aov)

    # compute post-hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv="team_score", within="robot_type", subject="user_id", padjust="bonf", effsize="cohen")

    # find any pairs where the difference is significant with corrected p-value
    print(post_hoc)
    print()

    # print hesitation times
    print("Hesitation times")
    for robot_type in robot_types:
        print(f"{robot_type}: {df[df.robot_type == robot_type].hesitation_time.mean()}, {df[df.robot_type == robot_type].hesitation_time.std()}")
    aov = pg.rm_anova(dv="hesitation_time", within=["robot_type"], subject="user_id", data=df)
    print(aov)
    # compute post-hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv="hesitation_time", within="robot_type", subject="user_id", padjust="bonf", effsize="cohen")
    print(post_hoc)

def plot_obj_selection(user_ids):
    all_data = {user_id: load_user_data(user_id) for user_id in user_ids}

    # get robot's objective function for cbp rollouts (maybe for first k timesteps?)
    k = 50
    obj_to_idx = {"influence": 0, "courtesy": 1}
    cbp_data = [all_data[user_id]["cbp"] for user_id in user_ids]
    all_objectives = []
    for user_data in cbp_data:
        for game_data in user_data:
            objectives = game_data["r_objective"][:k]
            obj_idxs = [obj_to_idx[obj] for obj in objectives]
            all_objectives.append(obj_idxs)

def plot_traj(user_id, robot_types=None, game_idxs=None):
    user_data = load_user_data(user_id)
    if robot_types is None:
        robot_types = ["baseline", "baseline_belief", "cbp"]
    if game_idxs is None:
        game_idxs = [1]
    robot_colors = {"baseline": "brown", "baseline_belief": "purple", "cbp": "green"}
    robot_cmaps = {"baseline": "copper_r", "baseline_belief": "Purples", "cbp": "Greens"}
    
    goal_colors = ["#3A637B", "#C4A46B", "#FF5A00"]

    for robot in robot_types:
        # for game_idx in range(len(user_data[robot])):
        for game_idx in game_idxs:
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
                    break
                else:
                    xh_traj_i = xh_traj[:,:i+1]
                    xr_traj_i = xr_traj[:,:i+1]
                overlay_timesteps(ax, xh_traj_i, xr_traj_i, n_steps=50, r_cmap=robot_cmaps[robot])
                ax.scatter(xh_traj[0,i], xh_traj[2,i], c="blue", s=150)
                ax.scatter(xr_traj[0,i], xr_traj[2,i], c=robot_colors[robot], s=150)
                goals_i = goals[:,:,i]
                ax.scatter(goals_i[0], goals_i[2], c=goal_colors, s=150, marker="x")
                ax.set_xlim(-11, 11)
                ax.set_ylim(-11, 8.5)
                ax.set_aspect("equal")

                # save images to make videos
                # filepath = f"./data/videos/user_study/frames_{user_id}_{robot}_{game_idx}/{i:03d}.png"
                # filepath = f"./data/videos/thesis_proposal/study_uncertain/{i:03d}.png"
                # os.makedirs(os.path.dirname(filepath), exist_ok=True)
                # plt.savefig(filepath, dpi=300)

                plt.pause(0.01)
            input(": ")

def plot_obj_selection(user_ids):
    all_data = {user_id: load_user_data(user_id) for user_id in user_ids}
    # get only cbp data
    cbp_data = {user_id: all_data[user_id]["cbp"] for user_id in user_ids}
    # TODO: complete

def load_survey_results(user_ids):
    baseline_file = "./data/user_study/survey_results/brown_survey.csv"
    baseline_belief_file = "./data/user_study/survey_results/purple_survey.csv"
    cbp_file = "./data/user_study/survey_results/green_survey.csv"
    robot_types = ["baseline", "baseline_belief", "cbp"]
    user_id = "[experimenter will fill out] Enter user ID number"
    baseline_df = pd.read_csv(baseline_file, index_col=user_id)
    baseline_belief_df = pd.read_csv(baseline_belief_file, index_col=user_id)
    cbp_df = pd.read_csv(cbp_file, index_col=user_id)
    robot_dfs = [baseline_df, baseline_belief_df, cbp_df]

    # get only the user ids we care about
    robot_dfs = [df.loc[user_ids] for df in robot_dfs]

    # add robot type to each df and join
    for idx, df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        df["robot_type"] = robot_type
    df = pd.concat(robot_dfs)

    questions = ["I felt like the robot and I scored as many points as we could.",
       "There were times when I changed my mind about which diamond to grab.",
       "I often changed which diamond I picked initially because of the robot. ",
       "The robot influenced me to pick good diamonds for the team.",
       "The robot's choice of diamonds made me choose worse diamonds for the team.",
       "I felt like the robot accounted for the diamond I wanted to pick when it was choosing a diamond.",
       "The robot was easy to collaborate with.",
       "I felt like the robot picked the best diamonds to grab for the team.",
       "I felt like the robot hindered the team's performance.",
       "[optional] What strategy did you and the robot use to collect diamonds?"]
    likert_scale = {"Strongly Agree": 5, "Agree": 4, "Neither agree nor disagree": 3, "Disagree": 2, "Strongly Disagree": 1}

    # map values in df to numeric
    for question in questions:
        df[question] = df[question].map(likert_scale)
    df = df.reset_index()

    question_names = ["many_points", "sometimes_change", "often_change", "good_influence", "bad_influence", "accounted_for", "easy_to_collaborate", "picked_best", "hindered_performance", "strategy"]

    good_influence = questions[3]
    bad_influence = questions[4]

    print("influence")
    # get responses for each question and convert to numeric
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        good_inf_q = robot_df[good_influence].map(likert_scale)
        bad_inf_q = 6 - robot_df[bad_influence].map(likert_scale) # reverse scale
        influence_q = (good_inf_q + bad_inf_q) / 2
        print(f"{robot_type}: {influence_q.mean()}, {influence_q.std()}")
        print(f"{robot_type}: {good_inf_q.mean()}, {good_inf_q.std()} good only")
    # convert df influence to average
    df["influence"] = (df[good_influence] + 6 - df[bad_influence]) / 2
    # run anova on influence question
    aov = pg.rm_anova(data=df, dv=good_influence, within="robot_type", subject=user_id)
    print(aov)
    # do post hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv=good_influence, within="robot_type", subject=user_id, padjust="bonf", effsize="cohen")
    print(post_hoc)
    print()

    print("bad influence")
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        bad_inf_q = robot_df[bad_influence].map(likert_scale)
        print(f"{robot_type}: {bad_inf_q.mean()}, {bad_inf_q.std()}")
    # run anova on bad influence question
    aov = pg.rm_anova(data=df, dv=bad_influence, within="robot_type", subject=user_id)
    print(aov)
    # do post hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv=bad_influence, within="robot_type", subject=user_id, padjust="bonf", effsize="cohen")
    print(post_hoc)
    print()

    print("accounted for")
    accounted_for = questions[5]
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        accounted_for_q = robot_df[accounted_for].map(likert_scale)
        print(f"{robot_type}: {accounted_for_q.mean()}, {accounted_for_q.std()}")
    # run anova on accounted for question
    aov = pg.rm_anova(data=df, dv=accounted_for, within="robot_type", subject=user_id)
    print(aov)
    # do post hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv=accounted_for, within="robot_type", subject=user_id, padjust="bonf", effsize="cohen")
    print(post_hoc)
    print()

    print("easy to collaborate")
    easy_to_collaborate = questions[6]
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        easy_to_collaborate_q = robot_df[easy_to_collaborate].map(likert_scale)
        print(f"{robot_type}: {easy_to_collaborate_q.mean()}, {easy_to_collaborate_q.std()}")
    # run anova on easy to collaborate question
    aov = pg.rm_anova(data=df, dv=easy_to_collaborate, within="robot_type", subject=user_id)
    print(aov)
    # do post hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv=easy_to_collaborate, within="robot_type", subject=user_id, padjust="bonf", effsize="cohen")
    print(post_hoc)
    print()

    print("picked best")
    picked_best = questions[7]
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        picked_best_q = robot_df[picked_best].map(likert_scale)
        print(f"{robot_type}: {picked_best_q.mean()}, {picked_best_q.std()}")
    # run anova on picked best question
    aov = pg.rm_anova(data=df, dv=picked_best, within="robot_type", subject=user_id)
    print(aov)
    # do post hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv=picked_best, within="robot_type", subject=user_id, padjust="bonf", effsize="cohen")
    print(post_hoc)
    print()

    print("hindered")
    hindered_performance = questions[8]
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        hindered_performance_q = robot_df[hindered_performance].map(likert_scale)
        print(f"{robot_type}: {hindered_performance_q.mean()}, {hindered_performance_q.std()}")
    # run anova on hindered performance question
    aov = pg.rm_anova(data=df, dv=hindered_performance, within="robot_type", subject=user_id)
    print(aov)
    # do post hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv=hindered_performance, within="robot_type", subject=user_id, padjust="bonf", effsize="cohen")
    print(post_hoc)
    print()

    print("many points")
    many_points = questions[0]
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        many_points_q = robot_df[many_points].map(likert_scale)
        print(f"{robot_type}: {many_points_q.mean()}, {many_points_q.std()}")
    # run anova on many points question
    aov = pg.rm_anova(data=df, dv=many_points, within="robot_type", subject=user_id)
    print(aov)
    # do post hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv=many_points, within="robot_type", subject=user_id, padjust="bonf", effsize="cohen")
    print(post_hoc)
    print()

    print("sometimes change")
    sometimes_change = questions[1]
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        sometimes_change_q = robot_df[sometimes_change].map(likert_scale)
        print(f"{robot_type}: {sometimes_change_q.mean()}, {sometimes_change_q.std()}")
    print()

    print("often change")
    often_change = questions[2]
    for idx, robot_df in enumerate(robot_dfs):
        robot_type = robot_types[idx]
        often_change_q = robot_df[often_change].map(likert_scale)
        print(f"{robot_type}: {often_change_q.mean()}, {often_change_q.std()}")
    # run anova on often change question
    aov = pg.rm_anova(data=df, dv=often_change, within="robot_type", subject=user_id)
    print(aov)
    # do post hoc pairwise tests
    post_hoc = pg.pairwise_tests(data=df, dv=often_change, within="robot_type", subject=user_id, padjust="bonf", effsize="cohen")
    print(post_hoc)
    print()

    # load post survey results and compare rankings
    post_survey_file = "./data/user_study/survey_results/post_survey.csv"
    post_survey_df = pd.read_csv(post_survey_file, index_col=user_id)
    # get only user ids we care about
    post_survey_df = post_survey_df.loc[user_ids]
    post_survey_qs = ['Rank the robots in order of which you preferred working with. [Brown Robot]',
       'Rank the robots in order of which you preferred working with. [Green Robot]',
       'Rank the robots in order of which you preferred working with. [Purple Robot]']
    robots = ["baseline", "baseline_belief", "cbp"]
    baseline_robot = post_survey_qs[0]
    baseline_belief_robot = post_survey_qs[1]
    cbp_robot = post_survey_qs[2]

    # map values in df to numeric
    rank_values = {"1 [most preferred]": 1, "2": 2, "3 [least preferred]": 3}
    for question in post_survey_qs:
        post_survey_df[question] = post_survey_df[question].map(rank_values)

    for idx, robot_type in enumerate(robots):
        robot_df = post_survey_df[post_survey_qs[idx]]
        # find % of times robot was ranked 1
        per_first = np.count_nonzero(robot_df == 1) / len(robot_df)
        print(f"{robot_type}: {robot_df.mean()}, {per_first}")

    # perform ranked-choice voting (instant runoff)
    survey_votes = post_survey_df[post_survey_qs].to_numpy()
    baseline = Candidate("baseline")
    baseline_belief = Candidate("baseline_belief")
    cbp = Candidate("cbp")
    candidates = [baseline, cbp, baseline_belief]
    ballots = []
    for vote in survey_votes:
        ballot = Ballot(ranked_candidates=np.array(candidates)[vote-1])
        ballots.append(ballot)
    election_result = pyrankvote.instant_runoff_voting(candidates, ballots)
    print(election_result)


if __name__ == "__main__":
    # user_ids = [4525, 6600, 7998]
    user_ids = [8544, 7193, 1977, 5328, 4628, 9088, 8340, 1546, 5655, 6380, 4595, 3083, 3507, 9649, 8061, 3327, 8427, 5392, 3853, 7815, 8614]
    # process_data(user_ids)
    # plot_traj(user_ids[0])
    # plot_obj_selection(user_ids)

    load_survey_results(user_ids)
    # plot_traj(user_ids[2])
    # plot_traj(user_ids[15])

    # plot_traj(user_ids[15], robot_types=["cbp"]) # saved with game_idx = 1
    # plot_traj(user_ids[2], robot_types=["cbp"])
    # plot_traj(user_ids[4], robot_types=["cbp"], game_idxs=[2]) # for thesis proposal: maybe 12 game_idx 1 to show how CBP is flexible (i.e. human picks other goal, CBP decides courtesy)
    # game_idx = 2; user_ids[0] is a good one (robot initially tries to influence, but realizes person isnt influencable so picks closer goal)
    # game_idx = 2; user_ids[4] is a good second example -- robot clearly influences them to choose an efficient goal
