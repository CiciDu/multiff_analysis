
% ---- (D) PLOT POPULATION STATISTICS WITH PlotSessions ----
% PlotSessions calculates and plots:
% - GAM summaries (fraction tuned, tuning curves, varexp, LL)
% - Canonical-correlation summaries
% - OLS decoder accuracy, true-vs-pred scatter, and true-vs-pred traces
%
% Requirements:
% 1. Need to enable coupled GAM fitting (required for PlotSessions)
% 2. Add population to session (runs AnalysePopulation)
% 3. Call PlotSessions with session array
%
% Note: This will take some time as it fits coupled GAM models
addpath(genpath(pwd));

script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..');
addpath(genpath(project_root));

script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..');
out_dir = fullfile(project_root, 'data', 'processed_data');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
out_path = fullfile(out_dir, 'session_data.mat');


S2 = load(out_path);
s = S2.s;

prs = default_prs(s.monk_id, s.sess_id);
prs = setupGAMParameters(prs);
prs.fitGAM_coupled = true;
prs.GAM_varexp = true;


% Plot everything from processed population stats
save_dir = fullfile(project_root, 'figs', 'PlotSessions');
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
PlotSessions([s], 'units', 'GAM', prs, save_dir);
PlotSessions([s], 'units', 'CANONCORR', prs, save_dir);
fprintf('Saved PlotSessions outputs to: %s\n', save_dir);
