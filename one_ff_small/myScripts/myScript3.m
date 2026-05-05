

%% Run AnalysePopulation and save results
addpath(genpath(pwd));

setup_project();
load(fullfile('data', 'baseline1.mat'));


s = experiments.sessions(1);
prs = default_prs(s.monk_id, s.sess_id);
prs = setupGAMParameters(prs);

prs.fitGAM_coupled = true;
prs.GAM_varexp = true;


% % Set all variables to optional (0) for variance explained computation
% % This is required because variance explained needs to test models with each variable removed
if isfield(prs, 'GAM_varchoose')
    prs.GAM_varchoose = zeros(size(prs.GAM_varchoose));
end


% pctRunOnAll dbstop if error


%% If only do analysis on a subset of units
% Create a new population object
s.populations(end+1) = population();
% % Analyze only the first 2 units
% s.populations(end).AnalysePopulation(s.units(1:2), 'units', s.behaviours, s.lfps, prs);
% Or: Analyze all units
s.populations(end).AnalysePopulation(s.units, 'units', s.behaviours, s.lfps, prs);

disp('AFTER AnalysePopulation');

script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..');


%% Plot and save figures
fig_dir = fullfile(project_root, 'data', 'processed_data', 'figures');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

PlotSessions([s], 'units', 'GAM', prs, fig_dir);
PlotSessions([s], 'units', 'CANONCORR', prs, fig_dir);
fprintf('PlotSessions complete! Saved figures to: %s\n', fig_dir);



delete(gcp('nocreate'));


disp('AFTER AnalysePopulation');

p = gcp('nocreate');
if ~isempty(p)
    wait(p);
    delete(p);
end

out_dir = fullfile(project_root, 'data', 'processed_data');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
out_path = fullfile(out_dir, 'session_data.mat');
save(out_path, 's', '-v7.3');
disp('Saved session data');


%S2 = load(out_path);


disp('SCRIPT FINISHED SUCCESSFULLY');