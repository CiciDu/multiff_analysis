%% Feel like this is more like a placeholder file
addpath(genpath(pwd));


% (D) Population GAM analysis using PlotPopulation
% NOTE: PlotPopulation is for GAM population analysis, NOT for firing rate plots
% PlotPopulation currently only supports 'GAM' plot type (not 'rate_move', 'rate_targ', etc.)
% 
% Method 1: Direct call (requires AnalysePopulation first)
% 1. Enable coupled GAM fitting:
%    prs.fitGAM_coupled = true;
% 2. Run AnalysePopulation:
%    pop_stats = AnalysePopulation(s.units(1:4), behv.trials, behv.stats, [], prs, []);
% 3. Call PlotPopulation:
%    PlotPopulation(behv, pop_stats, 'GAM', prs);
% 
% Method 2: Using session method (simpler)
% 1. Enable coupled GAM fitting:
%    prs.fitGAM_coupled = true;
% 2. Add population to session:
%    s.AddPopulation('units', prs);  % or 'singleunit' or 'multiunit'
% 3. Call session PlotPopulation:
%    s.PlotPopulation([], 'GAM', prs);  % [] means all unit types
% 
% Example (uncomment to use Method 2):
% prs.fitGAM_coupled = false;
% s.AddPopulation('units', prs);
% s.PlotPopulation('singleunit', 'GAM', prs);

pop_stats = AnalysePopulation(s.units(1:4), behv.trials, behv.stats, [], prs, []);
% PlotPopulation(behv,units,'GAM',prs)
PlotPopulation(behv,pop_stats,'GAM',prs)


但这个PlotPopulation，能不能在myScript3里用呢