%% function to plot neural population data from all sessions
function PlotSessions(sessions, unit_type, plot_type, prs, save_dir)

    show_unit_labels = true;

    % Decide whether to save figures
    if nargin < 5 || isempty(save_dir)
        save_figs = false;
    else
        save_figs = true;
        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
    end

    nsessions = length(sessions);

    switch plot_type
        case 'GAM'
            %% gather all units of type unit_type from all sessions
            units = struct.empty();
            for i = 1:nsessions
                pop_idx = length(sessions(i).populations); % use most recent population
                units = [units sessions(i).populations(pop_idx).(unit_type).stats.trialtype.all.models.log.units];
            end
            nunits = length(units);

            %% population statistics
            bestmodelclass = [];
            nvars = length(units(1).Uncoupledmodel.class{1});
            for i = 1:nunits
                if ~isnan(units(i).Uncoupledmodel.bestmodel)
                    bestmodelclass = [bestmodelclass ; ...
                        units(i).Uncoupledmodel.class{units(i).Uncoupledmodel.bestmodel}];
                else
                    bestmodelclass = [bestmodelclass ; false(1,nvars)];
                end
            end

            frac_tuned = sum(bestmodelclass) / nunits;

            fig = figure; hold on;
            set(fig,'Position',[80 200 900 400]);
            errorbar(1:nvars, frac_tuned, ...
                sqrt(frac_tuned .* (1 - frac_tuned) / nunits), ...
                'o','MarkerFace','b','CapSize',0);
            xlabel('Task variable');
            ylabel('Fraction of tuned neurons');
            set(gca,'XTick',1:9, ...
                'XTickLabel',units(1).Uncoupledmodel.xname, ...
                'TickLabelInterpreter','none');
            axis([0 10 0 1]);

            for hl = 0:0.2:1
                line([0 10],[hl hl],'Color',[0.8 0.8 0.8]);
            end

            if save_figs
                exportgraphics(fig, fullfile(save_dir,'GAM_frac_tuned.png'), ...
                    'Resolution',300);
            end

            %% tuning functions
            ncols = 8;
            nrows = ceil(max(frac_tuned) * nunits / ncols);

            for i = 1:nvars
                fig = figure; hold on;
                set(fig,'Position',[80 200 900 1500]);
                count = 0;

                for j = 1:nunits
                    if bestmodelclass(j,i)
                        count = count + 1;

                        stim = units(j).Uncoupledmodel.x{i};
                        rate = units(j).Uncoupledmodel.marginaltunings{ ...
                            units(j).Uncoupledmodel.bestmodel}{i};

                        if isstruct(rate)
                            rate = rate.mean;
                        end

                        vartype = units(j).Uncoupledmodel.xtype{i};
                        subplot(nrows, ncols, count); hold on;
                        plot(stim, rate);
                        if show_unit_labels
                            text(0.02, 0.95, sprintf('%d', j-1), ...
                                'Units','normalized', ...
                                'FontSize',8, ...
                                'Color',[0.3 0.3 0.3], ...
                                'VerticalAlignment','top');
                        end

                        yl = get(gca,'ylim');
                        yl = [floor(yl(1)) ceil(yl(2))];
                        set(gca,'ylim',yl,'YTick',yl);

                        xl = get(gca,'xlim');
                        xl = [floor(xl(1)) ceil(xl(2))];
                        set(gca,'xlim',xl,'XTick',xl);

                        set(gca,'Fontsize',10);

                        if strcmp(vartype,'event')
                            set(gca,'xlim',[-0.5 0.5],'XTick',[-0.5 0.5]);
                            yl = get(gca,'ylim');
                            line([0 0], yl, 'Color','k');
                        end
                    end
                end

                if nunits > 0
                    s = sgtitle(strrep(sprintf('%02d  Tuning to %s', ...
                        i, prs.varlookup(num2str(units(1).Uncoupledmodel.xname{i}))), ...
                        '_','\_'));
                    set(s,'FontSize',12,'FontWeight','Bold');
                end

                if save_figs
                    fname = sprintf('%02d_GAM_tuning_%s.png', ...
                        i, units(1).Uncoupledmodel.xname{i});
                    exportgraphics(fig, fullfile(save_dir,fname), ...
                        'Resolution',300);
                end
            end

            %% tuning heatmaps: peak-normalized tuning functions of tuned neurons, sorted by peak feature
            for i = 1:nvars
                tuned_idx = find(bestmodelclass(:,i));
                if isempty(tuned_idx)
                    continue;
                end

                vartype = units(1).Uncoupledmodel.xtype{i};
                if strcmp(vartype, '2D')
                    continue; % skip 2D covariates for heatmap
                end

                stim_list = cell(numel(tuned_idx), 1);
                rate_list = cell(numel(tuned_idx), 1);
                for k = 1:numel(tuned_idx)
                    j = tuned_idx(k);
                    stim = units(j).Uncoupledmodel.x{i}(:);
                    rate = units(j).Uncoupledmodel.marginaltunings{ ...
                        units(j).Uncoupledmodel.bestmodel}{i};
                    if isstruct(rate)
                        rate = rate.mean(:);
                    else
                        rate = rate(:);
                    end
                    if numel(stim) ~= numel(rate) || numel(rate) < 2
                        continue;
                    end
                    stim_list{k} = stim;
                    rate_list{k} = rate;
                end

                valid = ~cellfun(@isempty, rate_list);
                stim_list = stim_list(valid);
                rate_list = rate_list(valid);
                if isempty(rate_list)
                    continue;
                end

                x_all = unique(cell2mat(stim_list));
                x_all = sort(x_all(~isnan(x_all)));
                if numel(x_all) < 2
                    x_grid = linspace(min(cellfun(@min, stim_list)), max(cellfun(@max, stim_list)), 50);
                else
                    x_grid = linspace(min(x_all), max(x_all), 80);
                end

                n_curves = numel(rate_list);
                curves = nan(n_curves, numel(x_grid));
                peak_feat = nan(n_curves, 1);

                curves_raw = nan(n_curves, numel(x_grid));
                for k = 1:n_curves
                    xk = stim_list{k};
                    rk = rate_list{k};
                    rk_interp = interp1(xk, rk, x_grid, 'linear', 'extrap');
                    rk_max = max(rk_interp);
                    curves_raw(k,:) = rk_interp;
                    if rk_max > 0
                        curves(k,:) = rk_interp / rk_max;
                        [~, pk] = max(rk_interp);
                        peak_feat(k) = x_grid(pk);
                    else
                        curves(k,:) = rk_interp;
                        peak_feat(k) = x_grid(1);
                    end
                end

                [~, sort_idx] = sort(peak_feat);
                curves_sorted = curves(sort_idx, :);
                curves_raw_sorted = curves_raw(sort_idx, :);

                fig = figure;
                set(fig, 'Position', [80 120 700 500]);

                subplot(2,1,1); hold on;
                mu_rate = nanmean(curves_raw_sorted, 1);
                se_rate = nanstd(curves_raw_sorted, [], 1) / sqrt(n_curves);
                try
                    shadedErrorBar(x_grid, mu_rate, se_rate, 'lineprops', {'Color', [0.2 0.5 0.8], 'LineWidth', 2});
                catch
                    plot(x_grid, mu_rate, '-', 'Color', [0.2 0.5 0.8], 'LineWidth', 2);
                end
                if strcmp(vartype, 'event')
                    yl = ylim; line([0 0], yl, 'Color', 'k');
                end
                ylabel('Firing rate (spk/s)');
                set(gca, 'XTickLabel', []);
                set(gca, 'FontSize', 10); box off;

                subplot(2,1,2); hold on;
                imagesc(x_grid, 1:n_curves, curves_sorted);
                set(gca, 'YDir', 'normal');
                if strcmp(vartype, 'event')
                    yl = ylim; line([0 0], yl, 'Color', 'w', 'LineWidth', 1);
                end
                colormap(gca, parula);
                caxis([0 1]);
                xlabel(units(1).Uncoupledmodel.xname{i}, 'Interpreter', 'none');
                ylabel('Neurons');
                set(gca, 'FontSize', 10); box off;

                if isfield(prs, 'varlookup')
                    varlab = prs.varlookup(num2str(units(1).Uncoupledmodel.xname{i}));
                else
                    varlab = units(1).Uncoupledmodel.xname{i};
                end
                sgtitle(strrep(sprintf('Tuning to %s (peak-norm, sorted by peak)', varlab), '_', '\_'));

                if save_figs
                    fname = sprintf('%02d_GAM_tuning_heatmap_%s.png', i, units(1).Uncoupledmodel.xname{i});
                    exportgraphics(fig, fullfile(save_dir, fname), 'Resolution', 300);
                end
            end

            %% variance explained - coupled vs uncoupled
            count = 0;
            for j = 1:nunits
                count = count + 1;
                if ~isnan(units(j).Uncoupledmodel.bestmodel)
                    varexp_uncoupled(count) = mean( ...
                        units(j).Uncoupledmodel.testFit{ ...
                        units(j).Uncoupledmodel.bestmodel}(:,2));
                else
                    varexp_uncoupled(count) = 0;
                end
                varexp_coupled(count) = mean(units(j).Coupledmodel.testFit(:,2));
            end

            fig = figure; hold on;
            [F,X,FLO,FUP] = ecdf(varexp_uncoupled);
            try
                h1 = shadedErrorBar(X,F,[F-FLO FUP-F],'lineprops','-b');
            catch
                h1 = plot(X,F,'-b','LineWidth',2);
            end

            [F,X,FLO,FUP] = ecdf(varexp_coupled);
            try
                h2 = shadedErrorBar(X,F,[F-FLO FUP-F],'lineprops','-r');
            catch
                h2 = plot(X,F,'-r','LineWidth',2);
            end

            axis([0 0.2 0 1]);
            line([0 0.2],[0.5 0.5],'Color','k','LineStyle','--');
            xlabel('Variance explained');
            ylabel('Cumulative fraction of neurons');
            set(gca,'Fontsize',10);

            if isstruct(h1) && isfield(h1,'mainLine')
                legend([h1.mainLine h2.mainLine], ...
                    {'Uncoupled model','Coupled model'}, ...
                    'Location','SE','Fontsize',10);
            else
                legend([h1 h2], ...
                    {'Uncoupled model','Coupled model'}, ...
                    'Location','SE','Fontsize',10);
            end

            if save_figs
                exportgraphics(fig, fullfile(save_dir,'GAM_varexp_ecdf.png'), ...
                    'Resolution',300);
            end

            %% coupled vs uncoupled log likelihood
            fig = figure; hold on;
            for j = 1:nunits
                if ~isnan(units(j).Uncoupledmodel.bestmodel)
                    LL_unc = mean(units(j).Uncoupledmodel.testFit{ ...
                        units(j).Uncoupledmodel.bestmodel}(:,3));
                    LL_coup = mean(units(j).Coupledmodel.testFit(:,3));
                    plot(LL_unc, LL_coup, '.k');
                end
            end

            plot(0.01:0.01:3, 0.01:0.01:3, '--k');
            set(gca,'XScale','log','YScale','log');
            axis([0.01 3 0.01 3]);
            xlabel('(Uncoupled model), bits/spike');
            ylabel('(Coupled model), bits/spike');
            title('Log Likelihood','Fontsize',12);
            set(gca,'Fontsize',10);

            if save_figs
                exportgraphics(fig, ...
                    fullfile(save_dir,'GAM_LL_coupled_vs_uncoupled.png'), ...
                    'Resolution',300);
            end

        case {'CANONCORR', 'POPULATION'}

            decodertype = prs.decodertype;
            if isfield(prs, 'readout_varname')
                readout_vars = prs.readout_varname;
            else
                readout_vars = {};
            end

            canoncorr_coeff = {};
            canoncorr_dim = nan(nsessions,1);
            decoder_corr = nan(nsessions, numel(readout_vars));
            decoder_true = cell(nsessions, numel(readout_vars));
            decoder_pred = cell(nsessions, numel(readout_vars));
            decoder_trials_true = cell(nsessions, numel(readout_vars));
            decoder_trials_pred = cell(nsessions, numel(readout_vars));

            for i = 1:nsessions
                pop_idx = length(sessions(i).populations); % use most recent population
                pop_stats = sessions(i).populations(pop_idx).(unit_type).stats;

                % Canonical-correlation stats are stored under trialtype.all
                if isfield(pop_stats,'trialtype') && isfield(pop_stats.trialtype,'all') && ...
                        ~isempty(pop_stats.trialtype.all) && ...
                        isfield(pop_stats.trialtype.all(1),'canoncorr')
                    cc = pop_stats.trialtype.all(1).canoncorr;
                    if isfield(cc,'coeff') && ~isempty(cc.coeff)
                        canoncorr_coeff{end+1} = cc.coeff(:)'; %#ok<AGROW>
                    end
                    if isfield(cc,'dimensionality') && ~isempty(cc.dimensionality)
                        canoncorr_dim(i) = cc.dimensionality;
                    end
                end

                % Decoder quality for each requested readout variable
                if isfield(pop_stats, decodertype)
                    for j = 1:numel(readout_vars)
                        vname = readout_vars{j};
                        if isfield(pop_stats.(decodertype), vname) && ...
                                isfield(pop_stats.(decodertype).(vname), 'corr')
                            decoder_corr(i,j) = pop_stats.(decodertype).(vname).corr;
                        end
                        if isfield(pop_stats.(decodertype), vname)
                            vstats = pop_stats.(decodertype).(vname);
                            if isfield(vstats, 'true') && isfield(vstats, 'pred')
                                decoder_true{i,j} = vstats.true(:);
                                decoder_pred{i,j} = vstats.pred(:);
                            end
                            if isfield(vstats, 'trials') && isfield(vstats.trials, 'true') && ...
                                    isfield(vstats.trials, 'pred')
                                decoder_trials_true{i,j} = vstats.trials.true(:);
                                decoder_trials_pred{i,j} = vstats.trials.pred(:);
                                if isempty(decoder_true{i,j}) || isempty(decoder_pred{i,j})
                                    decoder_true{i,j} = cell2mat(vstats.trials.true(:));
                                    decoder_pred{i,j} = cell2mat(vstats.trials.pred(:));
                                end
                            end
                        end
                    end
                end
            end

            %% canonical-correlation spectrum
            if ~isempty(canoncorr_coeff)
                maxD = max(cellfun(@numel, canoncorr_coeff));
                coeff_mat = nan(numel(canoncorr_coeff), maxD);
                for i = 1:numel(canoncorr_coeff)
                    coeff_mat(i,1:numel(canoncorr_coeff{i})) = canoncorr_coeff{i};
                end

                mu = nanmean(coeff_mat,1);
                se = nanstd(coeff_mat,[],1) ./ sqrt(sum(~isnan(coeff_mat),1));

                fig = figure; hold on;
                set(fig,'Position',[80 200 900 400]);
                errorbar(1:maxD, mu, se, '-o', ...
                    'Color',[0.1 0.4 0.8], 'MarkerFaceColor',[0.1 0.4 0.8], 'CapSize',0);
                xlabel('Canonical dimension');
                ylabel('Canonical correlation');
                title('Population canonical-correlation spectrum');
                axis tight;
                set(gca,'Fontsize',10);

                if save_figs
                    exportgraphics(fig, fullfile(save_dir,'POP_canoncorr_spectrum.png'), ...
                        'Resolution',300);
                end
            end

            %% canonical task dimensionality
            if any(~isnan(canoncorr_dim))
                fig = figure; hold on;
                set(fig,'Position',[80 200 500 400]);
                valid_dim = canoncorr_dim(~isnan(canoncorr_dim));
                scatter(ones(size(valid_dim)), valid_dim, 36, ...
                    'MarkerFaceColor',[0.5 0.5 0.5], 'MarkerEdgeColor','k');
                errorbar(1, mean(valid_dim), std(valid_dim)/sqrt(numel(valid_dim)), ...
                    'ok', 'MarkerFaceColor','k', 'CapSize',0, 'LineWidth',1.5);
                xlim([0.5 1.5]);
                set(gca,'XTick',1,'XTickLabel',{'Dimensionality'});
                ylabel('Canonical task dimensionality');
                title('Canonical dimensionality across sessions');
                set(gca,'Fontsize',10);

                if save_figs
                    exportgraphics(fig, fullfile(save_dir,'POP_canoncorr_dimensionality.png'), ...
                        'Resolution',300);
                end
            end

            %% decoder performance by variable
            if ~isempty(readout_vars) && any(~isnan(decoder_corr(:)))
                mu = nanmean(decoder_corr,1);
                se = nanstd(decoder_corr,[],1) ./ sqrt(sum(~isnan(decoder_corr),1));

                fig = figure; hold on;
                set(fig,'Position',[80 200 1100 420]);
                errorbar(1:numel(readout_vars), mu, se, 'ok', ...
                    'MarkerFaceColor','k', 'CapSize',0, 'LineWidth',1.5);
                for j = 1:numel(readout_vars)
                    xj = j + 0.1*(rand(nsessions,1)-0.5);
                    scatter(xj, decoder_corr(:,j), 20, ...
                        'MarkerFaceColor',[0.7 0.7 0.7], ...
                        'MarkerEdgeColor',[0.4 0.4 0.4]);
                end
                line([0 numel(readout_vars)+1],[0 0],'Color',[0.7 0.7 0.7],'LineStyle','--');
                xlim([0.5 numel(readout_vars)+0.5]);
                set(gca,'XTick',1:numel(readout_vars), ...
                    'XTickLabel',readout_vars, ...
                    'TickLabelInterpreter','none');
                xtickangle(45);
                ylabel(sprintf('%s corr(true,pred)', decodertype));
                title('Population decoder performance across sessions');
                set(gca,'Fontsize',10);

                if save_figs
                    exportgraphics(fig, fullfile(save_dir,'POP_decoder_corr_by_var.png'), ...
                        'Resolution',300);
                end
            end

            %% single-trial decoding figure (style similar to reference)
            if ~isempty(readout_vars)
                preferred_vars = {'v','w','d','phi','dv','dw'};
                use_var_idx = [];
                for k = 1:numel(preferred_vars)
                    idx = find(strcmp(readout_vars, preferred_vars{k}), 1, 'first');
                    if ~isempty(idx)
                        use_var_idx(end+1) = idx; %#ok<AGROW>
                    end
                end
                if isempty(use_var_idx)
                    use_var_idx = 1:min(numel(readout_vars),6);
                else
                    use_var_idx = use_var_idx(1:min(numel(use_var_idx),6));
                end

                % Keep only variables with trial-wise decoder output
                keep = false(size(use_var_idx));
                for k = 1:numel(use_var_idx)
                    j = use_var_idx(k);
                    ntr = 0;
                    for i = 1:nsessions
                        if ~isempty(decoder_trials_true{i,j}) && ~isempty(decoder_trials_pred{i,j})
                            ntr = ntr + numel(decoder_trials_true{i,j});
                        end
                    end
                    keep(k) = (ntr > 0);
                end
                use_var_idx = use_var_idx(keep);

                if ~isempty(use_var_idx)
                    nrows = numel(use_var_idx);
                    ncols = 6;
                    row_colors = [ ...
                        0.55 0.75 0.10; ...
                        0.75 0.85 0.45; ...
                        0.90 0.35 0.10; ...
                        0.90 0.65 0.55; ...
                        0.10 0.60 0.85; ...
                        0.60 0.80 0.90];

                    fig = figure;
                    set(fig,'Position',[80 120 1200 140 + 120*nrows], ...
                        'Color',[0.93 0.93 0.93]);

                    for r = 1:nrows
                        j = use_var_idx(r);
                        tr_true = {};
                        tr_pred = {};
                        for i = 1:nsessions
                            if ~isempty(decoder_trials_true{i,j}) && ~isempty(decoder_trials_pred{i,j})
                                tr_true = [tr_true; decoder_trials_true{i,j}(:)]; %#ok<AGROW>
                                tr_pred = [tr_pred; decoder_trials_pred{i,j}(:)]; %#ok<AGROW>
                            end
                        end
                        ntr = min(numel(tr_true), numel(tr_pred));
                        if ntr == 0
                            continue;
                        end

                        pick = unique(round(linspace(1, ntr, min(ncols, ntr))));
                        for c = 1:numel(pick)
                            ax = subplot(nrows, ncols, (r-1)*ncols + c); hold(ax,'on');
                            yy_true = tr_true{pick(c)}(:);
                            yy_pred = tr_pred{pick(c)}(:);
                            L = min(numel(yy_true), numel(yy_pred));
                            if L < 2
                                continue;
                            end
                            t = 1:L;
                            plot(ax, t, yy_true(1:L), '-', 'Color',[0.6 0.6 0.6], 'LineWidth',2);
                            plot(ax, t, yy_pred(1:L), '-', ...
                                'Color',row_colors(min(r,size(row_colors,1)),:), 'LineWidth',2);
                            axis(ax,'tight');
                            set(ax,'XTick',[],'YTick',[],'Box','off','Color','none');
                            if c == 1
                                ylabel(ax, readout_vars{j}, 'Interpreter','none', 'FontWeight','bold');
                            end
                        end
                    end
                    sgtitle('Single-trial decoding', 'FontSize',22, 'FontWeight','bold');

                    if save_figs
                        exportgraphics(fig, fullfile(save_dir,'POP_decoder_singletrial_grid.png'), ...
                            'Resolution',300);
                    end
                end
            end
    end
end
