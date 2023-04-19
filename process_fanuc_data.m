%% loading data
% addpath('data');

% check times for naive test
load('data/fanuc_data/ravi_naive_test4');
disp(times(end)-times(1));

%% computing averages
naive_times = [74 60+13 60+41 60+14];
proactive_times = [60+9 60+15 60+23 61 60+20];

m1 = mean(naive_times);
s1 = std(naive_times);
m2 = mean(proactive_times);
s2 = std(proactive_times);

% bar chart
means = [m1 m2];
stds = [s1 s2];


bar(1:2, means);
% hold on
% er = errorbar(1:2, means, stds, stds);
% er.Color = [0 0 0];
% er.LineStyle = 'none';
% hold off;

