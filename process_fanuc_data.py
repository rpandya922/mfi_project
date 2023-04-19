import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_data(file_path):
    data = loadmat(file_path)
    return data["times"], data["h_xyz_wrist_data"], data["r_xyz_ee_data"]

def compute_idle_time(times, h_pos, thresh=0.02):
    """
    Computes the amount of time that the human's velocity is below a threshold

    times: numpy array of shape (1, N) of times (in seconds since the start of the experiment)
    h_pos: numpy array of shape (3, N) where N is the number of data points and each point is (x, y, z) position of the human
    """
    h_vel = np.diff(h_pos, axis=1) / np.diff(times)
    # compute the norm of the velocity
    h_vel_norm = np.linalg.norm(h_vel, axis=0)
    # compute the time intervals
    time_intervals = np.diff(times).squeeze()
    # compute the idle time
    idle_time = np.sum(time_intervals[h_vel_norm < thresh])
    return idle_time

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    # plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], color=color)

if __name__ == "__main__":
    naive_times = [74, 60+13, 60+41, 60+14]
    proactive_times = [60+9, 60+15, 60+23, 61]

    # compute the average time and standard deviation for each condition then plot a bar chart
    naive_completion_mean = np.mean(naive_times)
    naive_std = np.std(naive_times)
    proactive_completion_mean = np.mean(proactive_times)
    proactive_std = np.std(proactive_times)

    naive_files = ["./data/fanuc_data/ravi_naive_test4.mat", "./data/fanuc_data/ravi_naive_test5.mat",
                   "./data/fanuc_data/ravi_naive_test6.mat", "./data/fanuc_data/ravi_naive_test7.mat"]
    proactive_files = ["./data/fanuc_data/ravi_proactive_test1.mat", "./data/fanuc_data/ravi_proactive_test2.mat",
                          "./data/fanuc_data/ravi_proactive_test5.mat", "./data/fanuc_data/ravi_proactive_test6.mat"]
    # compute the average idle time for each condition
    naive_idle_times = []
    for file in naive_files:
        times, h_pos, _ = load_data(file)
        idle_time = compute_idle_time(times, h_pos)
        naive_idle_times.append(idle_time)
    proactive_idle_times = []
    for file in proactive_files:
        times, h_pos, _ = load_data(file)
        idle_time = compute_idle_time(times, h_pos)
        proactive_idle_times.append(idle_time)
    # compute mean and standard deviation of idle times
    naive_idle_mean = np.mean(naive_idle_times)
    naive_idle_std = np.std(naive_idle_times)
    proactive_idle_mean = np.mean(proactive_idle_times)
    proactive_idle_std = np.std(proactive_idle_times)

    # # plot boxplot of completion times 
    # plt.boxplot([naive_times, proactive_times], positions=[1, 1.5])
    # plt.xticks([1, 1.5], ['Naive', 'Proactive'])
    # plt.ylabel('Completion Time (s)')
    # plt.title('Completion Time')

    # # plot boxplot of idle times on same plot
    # plt.boxplot([naive_idle_times, proactive_idle_times], positions=[3, 3.5])
    # plt.xticks([3, 3.5], ['Naive', 'Proactive'])
    # plt.ylabel('Human Idle Time (s)')
    # plt.show()

    data_proative = [proactive_times, proactive_idle_times]
    data_naive = [naive_times, naive_idle_times]

    bpl = plt.boxplot(data_proative, positions=np.array(range(len(data_proative)))*2.0-0.4)
    bpr = plt.boxplot(data_naive, positions=np.array(range(len(data_naive)))*2.0+0.4)
    set_box_color(bpl, '#557f2d')
    set_box_color(bpr, '#7f6d5f')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#557f2d', label='Proactive')
    plt.plot([], c='#7f6d5f', label='Naive')
    plt.legend()

    plt.xticks(range(0, len(data_proative) * 2, 2), ['Completion Time', 'Idle Time'])
    plt.show()

    # plot the bar chart of the average completion time
    # plt.bar([0, 1], [naive_mean, proactive_mean], yerr=[naive_std, proactive_std], align='center', alpha=0.5, ecolor='black', capsize=10)
    # plt.bar([0, 1], [naive_completion_mean, proactive_completion_mean], align='center', alpha=0.5)
    # plt.xticks([0, 1], ['Naive', 'Proactive'])
    # plt.ylabel('Completion Time (s)')
    # plt.title('Average Time to Complete Task')

    # # make a bar chart of the average idle time
    # # plt.bar([0, 1], [naive_idle_mean, proactive_idle_mean], yerr=[naive_idle_std, proactive_idle_std], align='center', alpha=0.5, ecolor='black', capsize=10)
    # plt.bar([0, 1], [naive_idle_mean, proactive_idle_mean], align='center', alpha=0.5)
    # plt.xticks([0, 1], ['Naive', 'Proactive'])
    # plt.ylabel('Human Idle Time (s)')
    # plt.title('Average Human Idle Time')

    # make a grouped bar chart of the average completion time and average idle time
    # set width of bar
    # barWidth = 0.25
    # # set height of bar
    # bars1 = [naive_completion_mean, proactive_completion_mean]
    # bars2 = [naive_idle_mean, proactive_idle_mean]
    # # Set position of bar on X axis
    # r1 = np.arange(len(bars1))
    # r2 = [x + barWidth for x in r1]
    # # Make the plot
    # plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Average Completion Time')
    # plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Average Idle Time')
    # # add black dotted lines to compare the average completion and idle times
    # # plt.axhline(y=naive_completion_mean, color='black', linestyle='dashed')
    # plt.axhline(y=proactive_completion_mean, color='black', linestyle='dashed')
    # # plt.axhline(y=naive_idle_mean, color='black', linestyle='dashed')
    # plt.axhline(y=proactive_idle_mean, color='black', linestyle='dashed')

    # # Add xticks on the middle of the group bars
    # plt.xlabel('Robot Goal Selection Policy', fontweight='bold')
    # plt.xticks([r + barWidth/2 for r in range(len(bars1))], ['Naive', 'Proactive'])
    # # add y label
    # plt.ylabel('Time (s)')
    # # Create legend in top middle & Show graphic
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # make a bar chart instead grouped by completion time and idle time where proactive bars are the same color and naive bars are the same color
    # set width of bar
    barWidth = 0.25
    # set height of bar
    bars1 = [naive_completion_mean, naive_idle_mean]
    bars2 = [proactive_completion_mean, proactive_idle_mean]
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Naive')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Proactive')
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], ['Completion Time', 'Idle Time'])
    # add y label
    plt.ylabel('Time (s)')
    # Create legend in top middle & Show graphic
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    plt.show()
