function stats = AnalysePopulation(units,trials_behv,behv_stats,prs,stats)

nunits = length(units);
dt = prs.dt; % sampling resolution (s)

%% which analayses to do
compute_canoncorr = prs.compute_canoncorr;
regress_popreadout = prs.regress_popreadout;
simulate_population = prs.simulate_population;
corr_neuronbehverr = prs.corr_neuronbehverr;

%% load cases
trialtypes = fields(behv_stats.trialtype);
events = cell2mat({trials_behv.events});
continuous = cell2mat({trials_behv.continuous});

%% define filter to smooth the firing rate
t = linspace(-2*prs.filtwidth,2*prs.filtwidth,4*prs.filtwidth + 1);
h = exp(-t.^2/(2*prs.filtwidth^2));
h = h/sum(h);


%% cannonical correlation analysis
if compute_canoncorr
    varname = prs.canoncorr_varname;
    filtwidth = prs.neuralfiltwidth;
    for i=1% if i=1, fit model using data from all trials rather than separately to data from each condition
        nconds = length(behv_stats.trialtype.(trialtypes{i}));
        if ~strcmp((trialtypes{i}),'all') && nconds==1, copystats = true; else, copystats = false; end % only one condition means variable was not manipulated
        fprintf(['.........computing canonical correlations :: trialtype: ' (trialtypes{i}) '\n']);
        for j=1:nconds
            stats.trialtype.(trialtypes{i})(j).canoncorr.vars = varname;
            if copystats % if only one condition present, no need to recompute stats --- simply copy them from 'all' trials
                stats.trialtype.(trialtypes{i})(j).canoncorr = stats.trialtype.all.canoncorr;
            else
                trlindx = behv_stats.trialtype.(trialtypes{i})(j).trlindx;
                events_temp = events(trlindx);
                continuous_temp = continuous(trlindx);
                %% select variables of interest and load their details
                vars = cell(length(varname),1);
                for k=1:length(varname)
                    if isfield(continuous_temp,varname{k}), vars{k} = {continuous_temp.(varname{k})};
                    elseif isfield(behv_stats.pos_rel,varname{k}), vars{k} = behv_stats.pos_rel.(varname{k})(trlindx);
                    elseif strcmp(varname{k},'d')
                        vars{k} = cellfun(@(x,y) [zeros(sum(y<=0),1) ; cumsum(x(y>0)*dt)],{continuous_temp.v},{continuous_temp.ts},'UniformOutput',false);
                    elseif strcmp(varname{k},'dv')
                        vars{k} = cellfun(@(x) [0 ; diff(x)/dt],{continuous_temp.v},'UniformOutput',false);
                    elseif strcmp(varname{k},'dw')
                        vars{k} = cellfun(@(x) [0 ; diff(x)/dt],{continuous_temp.w},'UniformOutput',false);
                    elseif strcmp(varname{k},'phi')
                        vars{k} = cellfun(@(x,y) [zeros(sum(y<=0),1) ; cumsum(x(y>0)*dt)],{continuous_temp.w},{continuous_temp.ts},'UniformOutput',false);
                    elseif strcmp(varname(k),'eye_ver')
                        isnan_le = all(isnan(cell2mat({continuous_temp.zle}'))); isnan_re = all(isnan(cell2mat({continuous_temp.zre}')));
                        if isnan_le, vars{k} = {continuous_temp.zre};
                        elseif isnan_re, vars{k} = {continuous_temp.zle};
                        else, vars{k} = cellfun(@(x,y) 0.5*(x + y),{continuous_temp.zle},{continuous_temp.zre},'UniformOutput',false);
                        end
                    elseif strcmp(varname(k),'eye_hor')
                        isnan_le = all(isnan(cell2mat({continuous_temp.yle}'))); isnan_re = all(isnan(cell2mat({continuous_temp.yre}')));
                        if isnan_le, vars{k} = {continuous_temp.yre};
                        elseif isnan_re, vars{k} = {continuous_temp.yle};
                        else, vars{k} = cellfun(@(x,y) 0.5*(x + y),{continuous_temp.yle},{continuous_temp.yre},'UniformOutput',false);
                        end
                    end
                end
                %% define time windows for computing tuning
                timewindow_path = [[events_temp.t_targ]' [events_temp.t_stop]']; % when the subject is integrating path
                timewindow_full = [min([events_temp.t_move],[events_temp.t_targ]) - prs.pretrial ;... % from "min(move,targ) - pretrial_buffer"
                    [events_temp.t_end] + prs.posttrial]'; % till "end + posttrial_buffer"
                %% concatenate stimulus data from all trials
                trials_spks_temp = units(1).trials(trlindx);
                xt = [];
                for k=1:length(vars)
                    xt(:,k) = ConcatenateTrials(vars{k},[],{trials_spks_temp.tspk},{continuous_temp.ts},timewindow_full);
                    xt(isnan(xt(:,k)),k) = 0;
                end
                %% concatenate units
                Yt = zeros(size(xt,1),nunits);
                for k=1:nunits
                    trials_spks_temp = units(k).trials(trlindx);
                    [~,~,Yt(:,k)] = ConcatenateTrials(vars{1},[],{trials_spks_temp.tspk},{continuous_temp.ts},timewindow_full);
                end
                Yt_smooth = SmoothSpikes(Yt, 1*filtwidth);
                %% compute canonical correlation
                [X,Y,R,~,~,pstats] = canoncorr(xt,Yt_smooth/dt);
                stats.trialtype.(trialtypes{i})(j).canoncorr.stim = X;
                stats.trialtype.(trialtypes{i})(j).canoncorr.resp = Y;
                stats.trialtype.(trialtypes{i})(j).canoncorr.coeff = R;
                stats.trialtype.(trialtypes{i})(j).canoncorr.pval = pstats;
                %% canonical task dimensionality
                Dmax = numel(R); taskcov = zeros(1,Dmax);
                for k=1:Dmax
                    wx = X(:,k)/sqrt(X(:,k)'*X(:,k));
                    wy = Y(:,k)/sqrt(Y(:,k)'*Y(:,k));
                    xt_proj = xt*wx; yt_proj = (Yt_smooth/dt)*wy;
                    cov_temp = cov([xt_proj yt_proj]);
                    taskcov(k) = cov_temp(1,2); % off-diagonal entry                    
                end
                stats.trialtype.(trialtypes{i})(j).canoncorr.dimensionality = sum(taskcov)^2/sum(taskcov.^2); % defined analogously to participation ratio
                %% compute pairwise correlations
                stats.trialtype.(trialtypes{i})(j).responsecorr_raw = corr(Yt);
                stats.trialtype.(trialtypes{i})(j).responsecorr_smooth = corr(Yt_smooth);
                %% compute pairwise noise correlations
%                 for k=1:numel(varname)
%                     binrange{k} = prs.binrange.(varname{k}); 
%                     nbins{k} = prs.GAM_nbins{strcmp(prs.GAM_varname,varname{k})}; 
%                 end
%                 [stats.trialtype.(trialtypes{i})(j).noisecorr_raw,...
%                     stats.trialtype.(trialtypes{i})(j).noisecorr_smooth] = ComputeNoisecorr(Yt,xt,binrange,nbins);
            end
        end
    end
end

%% compute population readout weights via ordinary-least-squares
if regress_popreadout
    varname = prs.readout_varname;
    decodertype = prs.decodertype;
    filtwidth = prs.neuralfiltwidth;
    for i=1 % i=1 means compute only for all trials together
        nconds = length(behv_stats.trialtype.(trialtypes{i}));
        for j=1:nconds
            trlindx = behv_stats.trialtype.(trialtypes{i})(j).trlindx;
            events_temp = events(trlindx);
            continuous_temp = continuous(trlindx);
            %% define time windows for decoding
            timewindow_move = [[events_temp.t_move]' [events_temp.t_stop]']; % only consider data when the subject is integrating path
            timewindow_path = [[events_temp.t_targ]' [events_temp.t_stop]']; % only consider data when the subject is integrating path
            timewindow_full = [min([events_temp.t_move],[events_temp.t_targ]) - prs.pretrial ;... % from "min(move,targ) - pretrial_buffer"
                    [events_temp.t_end] + prs.posttrial]'; % till "end + posttrial_buffer"
            nunits = length(units);
            %% gather spikes from all units
            Yt = [];
            for k=1:nunits
                trials_spks_temp = units(k).trials(trlindx);
                [~,~,Yt(:,k)] = ConcatenateTrials({continuous_temp.v},[],{trials_spks_temp.tspk},{continuous_temp.ts},timewindow_full);
            end
            %% build decoder for each variable
            vars = cell(length(varname),1);
            trials_spks_temp = units(1).trials(trlindx);
            for k=1:length(varname)
                fprintf(['Building decoder for ' varname{k} '...\n']);
                if isfield(continuous_temp,varname{k}), vars{k} = {continuous_temp.(varname{k})};
                elseif isfield(behv_stats.pos_rel,varname{k}), vars{k} = behv_stats.pos_rel.(varname{k})(trlindx);
                elseif strcmp(varname{k},'d')
                    vars{k} = cellfun(@(x,y) [zeros(sum(y<=0),1) ; cumsum(x(y>0)*dt)],{continuous_temp.v},{continuous_temp.ts},'UniformOutput',false);
                elseif strcmp(varname{k},'dv')
                    vars{k} = cellfun(@(x) [0 ; diff(x)/dt],{continuous_temp.v},'UniformOutput',false);
                elseif strcmp(varname{k},'dw')
                    vars{k} = cellfun(@(x) [0 ; diff(x)/dt],{continuous_temp.w},'UniformOutput',false);
                elseif strcmp(varname{k},'phi')
                    vars{k} = cellfun(@(x,y) [zeros(sum(y<=0),1) ; cumsum(x(y>0)*dt)],{continuous_temp.w},{continuous_temp.ts},'UniformOutput',false);
                elseif strcmp(varname(k),'eye_ver')
                    isnan_le = all(isnan(cell2mat({continuous_temp.zle}'))); isnan_re = all(isnan(cell2mat({continuous_temp.zre}')));
                    if isnan_le, vars{k} = {continuous_temp.zre};
                    elseif isnan_re, vars{k} = {continuous_temp.zle};
                    else, vars{k} = cellfun(@(x,y) 0.5*(x + y),{continuous_temp.zle},{continuous_temp.zre},'UniformOutput',false);
                    end
                elseif strcmp(varname(k),'eye_hor')
                    isnan_le = all(isnan(cell2mat({continuous_temp.yle}'))); isnan_re = all(isnan(cell2mat({continuous_temp.yre}')));
                    if isnan_le, vars{k} = {continuous_temp.yre};
                    elseif isnan_re, vars{k} = {continuous_temp.yle};
                    else, vars{k} = cellfun(@(x,y) 0.5*(x + y),{continuous_temp.yle},{continuous_temp.yre},'UniformOutput',false);
                    end
                end
                xt = ConcatenateTrials(vars{k},[],{trials_spks_temp.tspk},{continuous_temp.ts},timewindow_full);
                % filter dv and dw
                t = linspace(-2*filtwidth,2*filtwidth,4*filtwidth + 1); h = exp(-t.^2/(2*filtwidth^2)); h = h/sum(h);
                if any(strcmp(varname{k},{'dv','dw'})), xt = conv(xt,h,'same'); end
                xt(isnan(xt)) = 0;
                %% fit smoothing window
                if prs.lineardecoder_fitkernelwidth % optimise filter width
                    filtwidths=1:5:100; decodingerror = nan(1,length(filtwidths));
                    fprintf('...optimising hyperparameter\n');
                    for l=1:length(filtwidths)
                        Yt_temp = SmoothSpikes(Yt, filtwidths(l)); % smooth spiketrains before fitting model
                        wts = (Yt_temp'*Yt_temp)\(Yt_temp'*xt); % analytical
                        decodingerror(l) = sqrt(sum((Yt_temp*wts - xt).^2));
                    end
                    [~,bestindx] = min(decodingerror); bestfiltwidth = filtwidths(bestindx);
                    stats.(decodertype).(varname{k}).bestfiltwidth = bestfiltwidth;
                    fprintf('**********decoding**********\n');
                    Yt_temp = SmoothSpikes(Yt, bestfiltwidth); % smooth spiketrains before fitting model
                    stats.(decodertype).(varname{k}).wts = (Yt_temp'*Yt_temp)\(Yt_temp'*xt); % analytical
                    stats.(decodertype).(varname{k}).true = xt;
                    stats.(decodertype).(varname{k}).pred = (Yt_temp*stats.(decodertype).(varname{k}).wts);
                    stats.(decodertype).(varname{k}).corr = corr(stats.(decodertype).(varname{k}).true,stats.(decodertype).(varname{k}).pred);
                else % use fixed filter width
                    Yt_temp = SmoothSpikes([Yt Zt], 5*filtwidth); % smooth spiketrains before fitting model
                    stats.(decodertype).(varname{k}).wts = (Yt_temp'*Yt_temp)\(Yt_temp'*xt); % analytical
                    stats.(decodertype).(varname{k}).true = xt;
                    stats.(decodertype).(varname{k}).pred = (Yt_temp*stats.(decodertype).(varname{k}).wts);
                    stats.(decodertype).(varname{k}).corr = corr(stats.(decodertype).(varname{k}).true,stats.(decodertype).(varname{k}).pred);
                end
                %% split back data into trials
                stats.(decodertype).(varname{k}).trials.true = DeconcatenateTrials(stats.(decodertype).(varname{k}).true,{continuous_temp.ts},timewindow_full);
                stats.(decodertype).(varname{k}).trials.pred = DeconcatenateTrials(stats.(decodertype).(varname{k}).pred,{continuous_temp.ts},timewindow_full);
                %% subsample neurons
                if prs.lineardecoder_subsample
                    N_neurons = prs.N_neurons; N_samples = prs.N_neuralsamples; Nt = size(Yt,1);
                    for l=1:numel(N_neurons)
                        fprintf(['.........decoding ' num2str(N_neurons(l)) ' neuron(s) \n']);
                        if N_neurons(l)< nunits, sampleindx = cell2mat(arrayfun(@(x) randperm(nunits,N_neurons(l))',1:N_samples,'UniformOutput',false));
                        else, sampleindx = [repmat((1:nunits)',1,N_samples) ; randi(nunits,[N_neurons(l)-nunits N_samples])]; end
                        Yt_temp = reshape(Yt(:,sampleindx),[Nt N_neurons(l) N_samples]);
                        Yt_temp = SmoothSpikes(Yt_temp, stats.(decodertype).(varname{k}).bestfiltwidth);
                        for m=1:N_samples
                            stats.(decodertype).(varname{k}).corr_subsample(l,m) = ...
                                corr(xt,squeeze(Yt_temp(:,:,m))*((squeeze(Yt_temp(:,:,m))'*squeeze(Yt_temp(:,:,m)))\(squeeze(Yt_temp(:,:,m))'*xt)));
                            stats.(decodertype).(varname{k}).popsize_subsample(l,m) = N_neurons(l);
                        end
                    end
                end
            end
            %% reconstruct trajectory from neural data
            for l = 1:sum(trlindx)
                [stats.(decodertype).xt_from_vw.trials.pred{l},stats.(decodertype).yt_from_vw.trials.pred{l}] = ...
                    gen_traj(stats.lineardecoder.w.trials.pred{l},stats.lineardecoder.v.trials.pred{l},continuous_temp(l).ts);
                stats.(decodertype).xt_from_vw.trials.pred{l}(end) = []; % remove extra time point at the end
                stats.(decodertype).yt_from_vw.trials.pred{l}(end) = [];
                [stats.(decodertype).xt.trials.pred{l},stats.(decodertype).yt.trials.pred{l}] = ...
                    gen_traj(diff(stats.lineardecoder.phi.trials.pred{l})/dt,diff(stats.lineardecoder.d.trials.pred{l})/dt,continuous_temp(l).ts);
            end
        end
    end
end

%%
if corr_neuronbehverr
    for k=1:sum(trlindx)
        startingtime = find(continuous_temp(k).ts > 0,1);
        stoppingtime = find(continuous_temp(k).ts > events_temp(k).t_stop,1);
        stats.error_lineardecoder.r_targ(k) = stats.lineardecoder.r_targ.trials.true{k}(stoppingtime) - stats.lineardecoder.r_targ.trials.pred{k}(stoppingtime);
        stats.error_lineardecoder.theta_targ(k) = stats.lineardecoder.theta_targ.trials.true{k}(stoppingtime) - stats.lineardecoder.theta_targ.trials.pred{k}(stoppingtime);
        stats.error_lineardecoder.v(k) = mean(stats.lineardecoder.v.trials.true{k}(startingtime:stoppingtime) - ...
            stats.lineardecoder.v.trials.pred{k}(startingtime:stoppingtime));
        stats.error_lineardecoder.w(k) = mean(stats.lineardecoder.w.trials.true{k}(startingtime:stoppingtime) - ...
            stats.lineardecoder.w.trials.pred{k}(startingtime:stoppingtime));
    end
    stats.error_behv.r_targ = behv_stats.trialtype.all.trlerrors;
end

%% evaluate model responses for coupled and uncoupled models
if simulate_population && exist('stats','var') && isfield(stats.trialtype.all,'models')
    varname = prs.simulate_varname;
    vartype = prs.simulate_vartype;
    filtwidth = prs.neuralfiltwidth;
    for i=1% if i=1, fit model using data from all trials rather than separately to data from each condition
        nconds = length(behv_stats.trialtype.(trialtypes{i}));
        if ~strcmp((trialtypes{i}),'all') && nconds==1, copystats = true; else, copystats = false; end % only one condition means variable was not manipulated
        fprintf(['.........computing canonical correlations for actual and model-simulated responses :: trialtype: ' (trialtypes{i}) '\n']);
        for j=1:nconds
            if copystats % if only one condition present, no need to recompute stats --- simply copy them from 'all' trials
                stats.trialtype.(trialtypes{i})(j).canoncorr = stats.trialtype.all.canoncorr;
            else
                trlindx = behv_stats.trialtype.(trialtypes{i})(j).trlindx;
                events_temp = events(trlindx);
                continuous_temp = continuous(trlindx);
                %% select variables of interest and load their details
                vars = cell(length(varname),1); binrange = cell(length(varname),1); nbins = cell(length(varname),1);
                for k=1:length(varname)
                    if isfield(continuous_temp,varname{k}), vars{k} = {continuous_temp.(varname{k})};
                    elseif isfield(behv_stats.pos_rel,varname{k}), vars{k} = behv_stats.pos_rel.(varname{k})(trlindx);
                    elseif strcmp(varname{k},'d')
                        vars{k} = cellfun(@(x,y) [zeros(sum(y<=0),1) ; cumsum(x(y>0)*dt)],{continuous_temp.v},{continuous_temp.ts},'UniformOutput',false);
                    elseif strcmp(varname{k},'phi')
                        vars{k} = cellfun(@(x,y) [zeros(sum(y<=0),1) ; cumsum(x(y>0)*dt)],{continuous_temp.w},{continuous_temp.ts},'UniformOutput',false);
                    elseif strcmp(varname(k),'eye_ver')
                        isnan_le = all(isnan(cell2mat({continuous_temp.zle}'))); isnan_re = all(isnan(cell2mat({continuous_temp.zre}')));
                        if isnan_le, vars{k} = {continuous_temp.zre};
                        elseif isnan_re, vars{k} = {continuous_temp.zle};
                        else, vars{k} = cellfun(@(x,y) 0.5*(x + y),{continuous_temp.zle},{continuous_temp.zre},'UniformOutput',false);
                        end
                    elseif strcmp(varname(k),'eye_hor')
                        isnan_le = all(isnan(cell2mat({continuous_temp.yle}'))); isnan_re = all(isnan(cell2mat({continuous_temp.yre}')));
                        if isnan_le, vars{k} = {continuous_temp.yre};
                        elseif isnan_re, vars{k} = {continuous_temp.yle};
                        else, vars{k} = cellfun(@(x,y) 0.5*(x + y),{continuous_temp.yle},{continuous_temp.yre},'UniformOutput',false);
                        end
                    end
                    binrange{k} = prs.binrange.(varname{k});
                    nbins{k} = prs.GAM_nbins{strcmp(prs.GAM_varname,varname{k})};
                end
                %% define time windows for computing tuning
                timewindow_path = [[events_temp.t_targ]' [events_temp.t_stop]']; % when the subject is integrating path
                timewindow_full = [min([events_temp.t_move],[events_temp.t_targ]) - prs.pretrial ;... % from "min(move,targ) - pretrial_buffer"
                    [events_temp.t_end] + prs.posttrial]'; % till "end + posttrial_buffer"
                %% concatenate stimulus data from all trials
                trials_spks_temp = units(1).trials(trlindx);
                xt = [];
                for k=1:length(vars)
                    xt(:,k) = ConcatenateTrials(vars{k},[],{trials_spks_temp.tspk},{continuous_temp.ts},timewindow_full); 
                    xt(isnan(xt(:,k)),k) = 0;
                end
                %% encode stimulus as one-hot variables
                x_1hot = [];
                for k=1:length(vars), x_1hot(:,:,k) = Encode1hot(xt,vartype{k},binrange{k},nbins{k}); end
                %% simulate uncoupled model
                Yt_uncoupled = zeros(size(xt,1),nunits);
                for k=1:nunits
                    y_temp = zeros(length(vars),size(xt,1));
                    uncoupledmodel = stats.trialtype.all.models.log.units(k).Uncoupledmodel;                    
                    if ~isnan(uncoupledmodel.bestmodel), wts = uncoupledmodel.wts{uncoupledmodel.bestmodel};
                    else, wts = uncoupledmodel.wts{1}; end
                    for l = 1:length(vars)
                        % simulated response to each variable before exp
                        y_temp(l,:) = sum(repmat(wts{strcmp(prs.GAM_varname,varname{l})},[size(x_1hot,1),1]).*squeeze(x_1hot(:,:,l)),2);
                    end
                    y_temp = exp(sum(y_temp));
                    Yt_uncoupled(:,k) = y_temp;
                end
                Yt_uncoupled_smooth = SmoothSpikes(Yt_uncoupled, 3*filtwidth);
                %% compute canonical correlation
                [X_uncoupled,Y_uncoupled,R_uncoupled,~,~,pstats_uncoupled] = canoncorr(xt,Yt_uncoupled_smooth/dt);
                stats.trialtype.(trialtypes{i})(j).canoncorr.uncoupled_stim = X_uncoupled;
                stats.trialtype.(trialtypes{i})(j).canoncorr.uncoupled_resp = Y_uncoupled;
                stats.trialtype.(trialtypes{i})(j).canoncorr.uncoupled_coeff = R_uncoupled;
                stats.trialtype.(trialtypes{i})(j).canoncorr.uncoupled_pval = pstats_uncoupled;
                %% canonical task dimensionality
                Dmax = numel(R_uncoupled); taskcov = zeros(1,Dmax);
                for k=1:Dmax
                    wx = X_uncoupled(:,k)/sqrt(X_uncoupled(:,k)'*X_uncoupled(:,k));
                    wy = Y_uncoupled(:,k)/sqrt(Y_uncoupled(:,k)'*Y_uncoupled(:,k));
                    xt_proj = xt*wx; yt_proj = (Yt_uncoupled_smooth/dt)*wy;
                    cov_temp = cov([xt_proj yt_proj]);
                    taskcov(k) = cov_temp(1,2); % off-diagonal entry
                end
                stats.trialtype.(trialtypes{i})(j).canoncorr.uncoupled_dimensionality = sum(taskcov)^2/sum(taskcov.^2); % defined analogously to participation ratio
                %% compute pairwise correlations
                stats.trialtype.(trialtypes{i})(j).responsecorr_uncoupled_raw = corr(Yt_uncoupled);
                stats.trialtype.(trialtypes{i})(j).responsecorr_uncoupled_smooth = corr(SmoothSpikes(Yt_uncoupled, filtwidth));
                %% compute pairwise noise correlations
                for k=1:numel(varname)
                    binrange{k} = prs.binrange.(varname{k});
                    nbins{k} = prs.GAM_nbins{strcmp(prs.GAM_varname,varname{k})};
                end
                [stats.trialtype.(trialtypes{i})(j).noisecorr_uncoupled_raw,...
                    stats.trialtype.(trialtypes{i})(j).noisecorr_uncoupled_smooth] = ComputeNoisecorr(Yt_uncoupled,xt,binrange,nbins);
                %% simulate coupled model
                Yt_coupled = zeros(size(xt,1),nunits);
                for k=1:nunits
                    y_temp = zeros(length(vars),size(xt,1));
                    coupledmodel = stats.trialtype.all.models.log.units(k).Coupledmodel;
                    wts = coupledmodel.wts;
                    for l = 1:length(vars)
                        % simulated response to each variable before exp
                        y_temp(l,:) = sum(repmat(wts{strcmp(prs.GAM_varname,varname{l})},[size(x_1hot,1),1]).*squeeze(x_1hot(:,:,l)),2);
                    end
                    Yt_coupled(:,k) = exp(sum(y_temp)' + ...
                        sum(Yt(:,1:nunits~=k).*repmat(stats.trialtype.all.models.log.units(k).Coupledmodel.wts{end},[size(Yt,1), 1]),2));
                end
                Yt_coupled_smooth = SmoothSpikes(Yt_coupled, 3*filtwidth);
                %% compute canonical correlation
                [X_coupled,Y_coupled,R_coupled,~,~,pstats_coupled] = canoncorr(xt,Yt_coupled_smooth/dt);
                stats.trialtype.(trialtypes{i})(j).canoncorr.coupled_stim = X_coupled;
                stats.trialtype.(trialtypes{i})(j).canoncorr.coupled_resp = Y_coupled;
                stats.trialtype.(trialtypes{i})(j).canoncorr.coupled_coeff = R_coupled;
                stats.trialtype.(trialtypes{i})(j).canoncorr.coupled_pval = pstats_coupled;
                %% canonical task dimensionality
                Dmax = numel(R_coupled); taskcov = zeros(1,Dmax);
                for k=1:Dmax
                    wx = X_coupled(:,k)/sqrt(X_coupled(:,k)'*X_coupled(:,k));
                    wy = Y_coupled(:,k)/sqrt(Y_coupled(:,k)'*Y_coupled(:,k));
                    xt_proj = xt*wx; yt_proj = (Yt_coupled_smooth/dt)*wy;
                    cov_temp = cov([xt_proj yt_proj]);
                    taskcov(k) = cov_temp(1,2); % off-diagonal entry
                end
                stats.trialtype.(trialtypes{i})(j).canoncorr.coupled_dimensionality = sum(taskcov)^2/sum(taskcov.^2); % defined analogously to participation ratio
                %% compute pairwise correlations
                stats.trialtype.(trialtypes{i})(j).responsecorr_coupled_raw = corr(Yt_coupled);
                stats.trialtype.(trialtypes{i})(j).responsecorr_coupled_smooth = corr(SmoothSpikes(Yt_coupled, filtwidth));
                %% compute pairwise noise correlations
                for k=1:numel(varname)
                    binrange{k} = prs.binrange.(varname{k});
                    nbins{k} = prs.GAM_nbins{strcmp(prs.GAM_varname,varname{k})};
                end
                [stats.trialtype.(trialtypes{i})(j).noisecorr_coupled_raw,...
                    stats.trialtype.(trialtypes{i})(j).noisecorr_coupled_smooth] = ComputeNoisecorr(Yt_coupled,xt,binrange,nbins);
            end
        end
    end
end

