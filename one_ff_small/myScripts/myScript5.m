% this is froms script3, mostly for debugging (check na in predictors

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
s.populations(end).AnalysePopulation(s.units(1:2), 'units', s.behaviours, s.lfps, prs);
% Or: Analyze all units
% s.populations(end).AnalysePopulation(s.units, 'units', s.behaviours, s.lfps, prs);

%% Plot
% PlotSessions([s], 'units', 'GAM', prs);
% fprintf('PlotSessions complete!\n');


