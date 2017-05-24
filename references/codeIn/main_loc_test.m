addpath('./IP_raytracing');
addpath('./Localization_algorithms');
%main_raytracing;
load offline_data_uniform.mat;
load online_data.mat;

rss = rss(1:1000, :);
trace = trace(1:1000, :);
predictions = online_loc( offline_rss, offline_location, rss, 'knn_reg');
acc = accuracy(predictions, trace);
fprintf('accuracy��%fm\n', acc / 100);
