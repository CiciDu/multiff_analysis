function prs = setupGAMParameters(prs, varargin)
% SETUPGAMPARAMETERS Configure parameters for GAM analysis
%
% Syntax:
%   prs = setupGAMParameters(prs)
%   prs = setupGAMParameters(prs, 'RemoveLFP', true)
%   prs = setupGAMParameters(prs, 'RemoveLFP', true, 'EvaluatePeaks', false)
%
% Inputs:
%   prs - Parameter structure from default_prs
%
% Optional Name-Value Pairs:
%   'RemoveLFP' - Remove LFP-related variables (default: true)
%   'EvaluatePeaks' - Enable peak evaluation (default: true)
%
% Output:
%   prs - Configured parameter structure
%
% Example:
%   prs = default_prs(monkey_id, session_id);
%   prs = setupGAMParameters(prs);

    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'RemoveLFP', true, @islogical);
    addParameter(p, 'EvaluatePeaks', true, @islogical);
    parse(p, varargin{:});
    
    % Enable GAM encoding
    prs.fitGAM_tuning = true;
    prs.GAM_varexp = false;

    
    % Add missing entries to varlookup for eye position variables
    if isfield(prs, 'varlookup')
        if ~isKey(prs.varlookup, 'eye_ver')
            prs.varlookup('eye_ver') = 'eye vert';
        end
        if ~isKey(prs.varlookup, 'eye_hor')
            prs.varlookup('eye_hor') = 'eye horiz';
        end
    end
    
    % Add missing entries to unitlookup for eye position variables
    if isfield(prs, 'unitlookup')
        if ~isKey(prs.unitlookup, 'eye_ver')
            prs.unitlookup('eye_ver') = 'deg';
        end
        if ~isKey(prs.unitlookup, 'eye_hor')
            prs.unitlookup('eye_hor') = 'deg';
        end
    end
    
    % Remove LFP-related variables if requested
    if p.Results.RemoveLFP
        % Create mask for non-LFP variables (only if GAM_varname exists)
        if isfield(prs, 'GAM_varname')
            lfp_mask = ~strcmp(prs.GAM_varname, 'phase');
            
            % Apply mask to GAM parameters (check each field exists)
            if isfield(prs, 'GAM_varname'), prs.GAM_varname = prs.GAM_varname(lfp_mask); end
            if isfield(prs, 'GAM_vartype'), prs.GAM_vartype = prs.GAM_vartype(lfp_mask); end
            if isfield(prs, 'GAM_basistype'), prs.GAM_basistype = prs.GAM_basistype(lfp_mask); end
            if isfield(prs, 'GAM_nbins'), prs.GAM_nbins = prs.GAM_nbins(lfp_mask); end
            if isfield(prs, 'GAM_lambda'), prs.GAM_lambda = prs.GAM_lambda(lfp_mask); end
            if isfield(prs, 'GAM_varchoose'), prs.GAM_varchoose = prs.GAM_varchoose(lfp_mask); end
            
            % Remove 'phase' from tuning variable lists (check each field exists)
            prs.GAM_varname = setdiff(prs.GAM_varname, {'phase'}, 'stable');
        end
        
        if isfield(prs, 'tuning_continuous')
            prs.tuning_continuous = setdiff(prs.tuning_continuous, {'phase'}, 'stable');
        end
        
        if isfield(prs, 'tuning_event')
            prs.tuning_event = setdiff(prs.tuning_event, {'phase'}, 'stable');
        end
        
        % Disable LFP-spike relation analysis (check each field exists)
        if isfield(prs, 'analyse_spikeLFPrelation')
            prs.analyse_spikeLFPrelation = false;
        end
        if isfield(prs, 'analyse_spikeLFPrelation_allLFPs')
            prs.analyse_spikeLFPrelation_allLFPs = false;
        end
    end
    
    % Configure peak evaluation
    prs.evaluate_peaks = p.Results.EvaluatePeaks;
    
end
