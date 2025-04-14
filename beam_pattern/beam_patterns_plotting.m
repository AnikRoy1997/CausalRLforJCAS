%% Plot beam patterns for multiple scenes
clc;
clear all;

num_scene = 1;  % greater than 1 for dynamic scenario
for i=1:num_scene
    load(['D:\My Academics\Ph.D\Ericsson_Internship\Codebook_Learning_RL_DeepMIMO\Codebook_Learning_RL-O2Sensing/beam_codebook_set/beam_codebook_' num2str(i-1) '.mat']);
    plot_pattern(beams.',i);
    saveas(gcf, ['D:\My Academics\Ph.D\Ericsson_Internship\Codebook_Learning_RL_DeepMIMO\Codebook_Learning_RL-O2Sensing/beam_plots/beam_plot_' num2str(i-1) '.png']);
end