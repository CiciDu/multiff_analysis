%% Save data in a Python-friendly format (SAFE, CLEAN, FINAL)

addpath(genpath(pwd))
setup_project();
load(fullfile('data', 'baseline1.mat'));

sessions_in  = experiments.sessions;
sessions_out = struct([]);

for i = 1:numel(sessions_in)

    s = sessions_in(i);

    % =========================
    % Session-level metadata
    % =========================
    sessions_out(i).monk_id   = s.monk_id;
    sessions_out(i).sess_id   = s.sess_id;
    sessions_out(i).sess_date = s.sess_date;
    sessions_out(i).trlindx   = s.behaviours.stats.trialtype.all.trlindx;

    % =========================
    % Behaviour (SAFE creation)
    % =========================
    sessions_out(i).behaviour = struct();
    sessions_out(i).behaviour.trials = struct([]);
    sessions_out(i).behaviour.stats = struct('pos_rel', []);

    bt = s.behaviours.trials;
    bs = s.behaviours.stats;

    for t = 1:numel(bt)

        % --- raw per-trial data ---
        sessions_out(i).behaviour.trials(t).continuous = bt(t).continuous;
        sessions_out(i).behaviour.trials(t).events     = bt(t).events;
        sessions_out(i).behaviour.trials(t).logical    = bt(t).logical;
        sessions_out(i).behaviour.trials(t).prs        = bt(t).prs;

        % --- per-trial behaviour stats (ONLY pos_rel, CONSISTENT STRUCT) ---
        stats_t = struct();
        stats_t.pos_rel = struct();   % <-- ALWAYS create this field
        
        if isfield(bs, 'pos_rel')
            fn = fieldnames(bs.pos_rel);
            for f = 1:numel(fn)
                fld = fn{f};
                stats_t.pos_rel.(fld) = slice_stat_field(bs.pos_rel.(fld), t);
            end
        end
        
        sessions_out(i).behaviour.stats(t) = stats_t;

    end

    % =========================
    % Units (SAFE creation)
    % =========================
    uin = s.units;
    sessions_out(i).units = struct([]);

    for k = 1:numel(uin)
        u = uin(k);

        sessions_out(i).units(k).cluster_id     = u.cluster_id;
        sessions_out(i).units(k).channel_id     = u.channel_id;
        sessions_out(i).units(k).electrode_id   = u.electrode_id;
        sessions_out(i).units(k).electrode_type = u.electrode_type;
        sessions_out(i).units(k).brain_area     = u.brain_area;
        sessions_out(i).units(k).spkwf          = u.spkwf;
        sessions_out(i).units(k).spkwidth       = u.spkwidth;
        sessions_out(i).units(k).type           = u.type;
        sessions_out(i).units(k).trials         = u.trials;
        sessions_out(i).units(k).stats          = u.stats;
    end
end

% =========================
% Save (TWO locations)
% =========================

% MultiFF canonical copy
out_dir = '/Users/dusiyi/Documents/Multifirefly-Project/all_monkey_data/one_ff_data';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
save(fullfile(out_dir, 'sessions_python.mat'), 'sessions_out', '-v6')

disp('Saved sessions_python.mat successfully.')

%% =========================
% Helper (must be at END)
% =========================
function val_t = slice_stat_field(val, t)
% Safely slice per-trial stats fields

    if isempty(val)
        val_t = val;

    % cell array: {t}
    elseif iscell(val)
        val_t = val{t};

    % numeric vector: (nTrials,)
    elseif isnumeric(val) && isvector(val)
        val_t = val(t);

    % numeric matrix: (T x nTrials) or (nTrials x T)
    elseif isnumeric(val) && ndims(val) == 2
        sz = size(val);
        if sz(2) >= t
            val_t = val(:, t);
        elseif sz(1) >= t
            val_t = val(t, :).';
        else
            val_t = val;
        end

    % fallback: metadata / scalar
    else
        val_t = val;
    end
end
