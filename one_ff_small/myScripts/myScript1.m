%% Show tuning of individual units
addpath(genpath(pwd));

setup_project();
load(fullfile('data', 'baseline1.mat'));



s = experiments.sessions(1);
prs = default_prs(s.monk_id, s.sess_id);
% Configure GAM parameters and remove LFP-related variables
prs = setupGAMParameters(prs);


s = experiments.sessions(1);
behv = s.behaviours;
trials_behv = s.behaviours.trials;
behv_stats  = s.behaviours.stats;


% ---- RUN ANALYSIS for some units----
% for i = 1:numel(s.units) % if using all units
num_units = 3;
for i = 1:num_units
    u = s.units(i);
    trials_spks = u.trials;
    trials_behv = behv.trials;
    behv_stats  = behv.stats;
    lfps = [];
    s.units(i).stats = AnalyseUnit(trials_spks, trials_behv, behv_stats, lfps, prs);
end


% ---- PLOT RESULTS ----
for i = 1:num_units
    conditions = s.units(i).stats.trialtype.all;
    nconds = numel(conditions);

    % collect union of all fields
    allfields = {};
    for k = 1:nconds
        allfields = union(allfields, fieldnames(conditions(k).GAM.log));
    end

    % initialize condition with empty fields
    condition = struct();
    for f = 1:numel(allfields)
        [condition(1:nconds).(allfields{f})] = deal([]);
    end

    % fill available fields
    for k = 1:nconds
        fn = fieldnames(conditions(k).GAM.log);
        for f = 1:numel(fn)
            condition(k).(fn{f}) = conditions(k).GAM.log.(fn{f});
        end
    end
    PlotUnit(prs, behv, condition, 'gam_uncoupled', 'all');
end

% (C) Population firing rates across units (requires prs.evaluate_peaks = true before AnalyseUnit)
PlotUnits(prs, behv, s.units(1:num_units), 'rate_move', 'all');
PlotUnits(prs, behv, s.units(1:num_units), 'rate_targ', 'all');

