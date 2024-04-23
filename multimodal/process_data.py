import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from scipy.spatial import HalfspaceIntersection

from simulate import overlay_timesteps

def get_r_goals_reached(data):
    goals_reached = []
    n_traj = len(data)
    for i in range(n_traj):
        goals_reached.append((np.array(data[i]["r_goal_reached"]) >= 0).sum())
    return goals_reached

def get_safety_violations(data, dmin=1.0, print_v=False):
    safety_violations = []
    n_traj = len(data)
    for i in range(n_traj):
        if print_v:
            print(np.where(np.array(data[i]["distances"]) < dmin))
        safety_violations.append((np.array(data[i]["distances"]) < dmin).sum())
    return safety_violations

def get_times_close(data, close=2.0, dmin=1.0):
    close_timesteps = []
    n_traj = len(data)
    for i in range(n_traj):
        times = np.where(np.array(data[i]["distances"]) < dmin)[0]
        if len(times) > 0:
            pass
        close_timesteps.append(np.where(np.array(data[i]["distances"]) < dmin)[0])
    import ipdb; ipdb.set_trace()

def get_control_space_size(data, umax=30):
    control_space_size = []
    n_traj = len(data)
     # generate meshgrid of controls
    us = np.linspace(-umax, umax, 100)
    U1, U2 = np.meshgrid(us, us)
    U = np.vstack((U1.flatten(), U2.flatten()))
    for i in tqdm(range(n_traj)):
        Ls = np.array(data[i]["all_Ls"])
        Ss = np.array(data[i]["all_Ss"])
        Ls = Ls[:,0,:]
        Ss = np.amin(Ss, axis=1) # smallest S will be hardest to satisfy Lu - S <= 0 (and constraints are parallel)

        # compute if each point satisfies constraints
        c_satisfied = ((Ls @ U) - Ss[:,None]) <= 0
        c_satisfied_num = np.sum(c_satisfied, axis=1)
        space_percent = c_satisfied_num / U.shape[1]
        
        # only count timesteps where safety constraint is active
        safety_active = np.array(data[i]["safety_actives"])
        space_percent = space_percent[safety_active]
        if len(space_percent) == 0:
            control_space_size.append(np.nan)
            continue
        control_space_size.append(np.mean(space_percent)*(umax**2))
    return control_space_size

def plot_performance(data):
    baseline = data["baseline"]
    multimodal = data["multimodal"]
    SEA = data["SEA"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes = axes.flatten()
    for i in range(len(baseline)):
        
        ax = axes[0]
        ax.clear()
        ax.plot(baseline[i]["distances"], label="baseline")
        ax.plot(multimodal[i]["distances"], label="multimodal")
        ax.plot(SEA[i]["distances"], label="SEA")
        ax.plot(np.ones_like(baseline[i]["distances"]), c="black", linestyle="--", label="min distance")
        ax.set_xlabel("timestep")
        ax.set_ylabel("distance b/w agents")
        ax.legend()

        # add a verical line for each time the human reaches a goal (SEA only)
        for j, r_goal_reached in enumerate(SEA[i]["h_goal_reached"]):
            if r_goal_reached >= 0:
                ax.axvline(x=j, c="red", linestyle="--", alpha=0.5)

        ax = axes[1]
        ax.clear()
        ax.plot(baseline[i]["r_goal_dists"], label="baseline")
        ax.plot(multimodal[i]["r_goal_dists"], label="multimodal")
        ax.plot(SEA[i]["r_goal_dists"], label="SEA")
        ax.set_xlabel("timestep")
        ax.set_ylabel("distance to goal")

        plt.pause(0.001)
        print(i)
        input(": ")

def plot_constraints(data):
    fig, ax = plt.subplots()
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_aspect("equal")
    all_Ls = np.array(data["all_Ls"])
    all_Ss = np.array(data["all_Ss"])

    # for i in range(200):
    i = 28
    ax.clear()
    Ls = all_Ls[i]
    Ss = all_Ss[i]

    halfspaces = np.hstack((Ls, -Ss[:,None]))

    x = np.linspace(-30, 30, 100)
    symbols = ['-', '+', 'x', '*']
    signs = [0, 0, 0]
    fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}
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
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
        # plt.pause(0.001)
        # print(i)
        # input(": ")

if __name__ == "__main__":
    # filename = "./data/sim_stats_20230804-184019.pkl" # big file w/ 1000 trajectories (used for camera ready draft on 3/15/24 at 10am)
    # filename = "./data/sim_stats_20230810-150707.pkl" # 100 traj used for paper draft results on 8/11/23
    # filename = "./data/sim_stats_20230810-201114.pkl"
    # filename = "./data/sim_stats_20240315-173925.pkl"
    filename = "./data/sim_stats_20240413-164419.pkl" # traj with follower robot on 4/13/24, eta=2, k_phi=10, dmin=3
    # filename = "./data/sim_stats_20240413-214340.pkl" # traj with normal robot on 4/13/24, eta=2, k_phi=5, dmin=1.5
    # filename = "./data/sim_stats_20240417-213542.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)

    # plot_performance(data)

    # found idx 16 as trajectory where only SEA breaks safe distance
    # with open("./data/traj_16.pkl", "wb") as f:
    #     pickle.dump({"baseline": data["baseline"][16], "SEA": data["SEA"][16], "multimodal": data["multimodal"][16]}, f)
    # plot_traj(data["SEA"][16])

    baseline = data["baseline"]
    multimodal = data["multimodal"]
    SEA = data["SEA"]
    # SEA = data["reference"]
    # plot_constraints(baseline[0])
    # plot_constraints(multimodal[0])
    # plt.show()

    goals_reached = {"baseline": [], "multimodal": [], "SEA": []}
    goals_reached["baseline"] = np.array(get_r_goals_reached(baseline))
    goals_reached["multimodal"] = np.array(get_r_goals_reached(multimodal))
    goals_reached["SEA"] = np.array(get_r_goals_reached(SEA))
    
    dmin = 3.0
    safety_violations = {"baseline": [], "multimodal": [], "SEA": []}
    safety_violations["baseline"] = np.array(get_safety_violations(baseline, dmin=dmin))
    safety_violations["multimodal"] = np.array(get_safety_violations(multimodal, dmin=dmin))
    safety_violations["SEA"] = np.array(get_safety_violations(SEA, dmin=dmin))
    n_traj = len(safety_violations["SEA"])

    # get_times_close(baseline, dmin=2.0)

    umax = 30
    control_space = {"baseline": [], "multimodal": [], "SEA": []}
    control_space["baseline"] = np.array(get_control_space_size(baseline, umax=umax))
    control_space["multimodal"] = np.array(get_control_space_size(multimodal, umax=umax))
    control_space["SEA"] = np.array(get_control_space_size(SEA, umax=umax))

    print("goals reached")
    print(f"baseline: {np.mean(goals_reached['baseline'])}, {np.std(goals_reached['baseline'])}")
    print(f"multimodal: {np.mean(goals_reached['multimodal'])}, {np.std(goals_reached['multimodal'])}")
    print(f"SEA: {np.mean(goals_reached['SEA'])}, {np.std(goals_reached['SEA'])}")
    # make data into dataframe, then run anova 
    df = pd.DataFrame(goals_reached)
    df_ = pd.melt(df, var_name="robot_type", value_name="team_score")
    df_["init_cond"] = df_.groupby(df_["robot_type"]).cumcount()
    aov = pg.rm_anova(dv="team_score", within=["robot_type"], subject="init_cond", data=df_, correction=True)
    print(aov)
    post_hoc = pg.pairwise_tests(data=df_, dv="team_score", within="robot_type", subject="init_cond", padjust="bonf", effsize="cohen")
    print(post_hoc)

    print()
    print("safety violations")
    print(f"baseline: {np.mean(safety_violations['baseline'])}")
    print(f"multimodal: {np.mean(safety_violations['multimodal'])}")
    print(f"SEA: {np.mean(safety_violations['SEA'])}")
    print()
    print("safety rate")
    traj_len = SEA[0]["xh_traj"].shape[1]
    print(f"baseline: {(1 - (safety_violations['baseline'].sum() / (traj_len*n_traj)))*100}")
    print(f"multimodal: {(1 - (safety_violations['multimodal'].sum() / (traj_len*n_traj)))*100}")
    print(f"SEA: {(1 - (safety_violations['SEA'].sum() / (traj_len*n_traj)))*100}")
    print()

    print("control space size")
    control_space_ = {}
    control_space_["baseline"] = control_space["baseline"][~np.isnan(control_space["baseline"])]
    control_space_["multimodal"] = control_space["multimodal"][~np.isnan(control_space["multimodal"])]
    control_space_["SEA"] = control_space["SEA"][~np.isnan(control_space["SEA"])]
    print(f"baseline: {np.mean(control_space_['baseline'])}, {np.std(control_space_['baseline'])}")
    print(f"multimodal: {np.mean(control_space_['multimodal'])}, {np.std(control_space_['multimodal'])}")
    print(f"SEA: {np.mean(control_space_['SEA'])}, {np.std(control_space_['SEA'])}")
    # make data into dataframe, then run anova 
    df = pd.DataFrame(control_space) # use original data with nans so we have same number of samples for each
    df_ = pd.melt(df, var_name="robot_type", value_name="team_score")
    df_["init_cond"] = df_.groupby(df_["robot_type"]).cumcount()
    aov = pg.rm_anova(dv="team_score", within=["robot_type"], subject="init_cond", data=df_, correction=True)
    print(aov)
    post_hoc = pg.pairwise_tests(data=df_, dv="team_score", within="robot_type", subject="init_cond", padjust="bonf", effsize="cohen")
    print(post_hoc)

    # make box and whisker plot
    fig, ax = plt.subplots()
    # ax.boxplot([goals_reached["baseline"], goals_reached["multimodal"], goals_reached["SEA"]], showmeans=True)
    # ax.set_xticklabels(["baseline", "multimodal", "SEA"])
    # ax.set_ylabel("goals reached")

    ax.boxplot([control_space_["baseline"], control_space_["multimodal"], control_space_["SEA"]], showmeans=True)
    ax.set_xticklabels(["baseline", "multimodal", "SEA"])
    ax.set_ylabel("control space size")
    plt.show()
