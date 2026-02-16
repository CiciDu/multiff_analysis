import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from neural_data_analysis.neural_analysis_tools.pgam_tools import pgam_utils
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import (
    plot_modeling_result
)

PGAM_PATH = Path(
    'multiff_analysis/external/pgam/src'
).expanduser().resolve()

if str(PGAM_PATH) not in sys.path:
    sys.path.append(str(PGAM_PATH))



from PGAM.GAM_library import *
from post_processing import postprocess_results
from sklearn.preprocessing import StandardScaler
import PGAM.gam_data_handlers as gdh

def find_project_root(marker="multiff_analysis"):
    """Search upward until we find a folder containing `marker`."""
    cur = Path(os.getcwd()).resolve()   # use CWD instead of __file__
    for parent in [cur] + list(cur.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find project root with marker '{marker}'")


project_root = find_project_root()

# Build the paths relative to project root
pgam_src = project_root / "multiff_analysis" / "external" / "pgam" / "src"
pgam_src_pg = pgam_src / "PGAM"

for path in [pgam_src, pgam_src_pg]:
    if str(path) not in sys.path:
        sys.path.append(str(path))


class LoadedModelData:
    """Simple class to hold loaded model data as attributes."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class PGAMclass():

    # temporal_vars = ['catching_ff', 'log1p_num_ff_visible', 'monkey_speeddummy',
    #                  'min_target_has_disappeared_for_last_time_dummy',
    #                  'min_target_cluster_has_disappeared_for_last_time_dummy',
    #                  'max_target_visible_dummy', 'max_target_cluster_visible_dummy',
    #                  'retry_capture_indice_dummy', 'retry_switch_indice_dummy',
    #                  'ignore_sudden_flash_indice_dummy', 'two_in_a_row', 'waste_cluster_around_target',
    #                  'visible_before_last_one', 'disappear_latest', 'ignore_sudden_flash',
    #                  'retry_capture', 'retry_switch', 'cluster_around_target',
    #                  ]

    temporal_vars = ['capture_ff', 'log1p_num_ff_visible', 'log1p_num_ff_in_memory', 'turning_right', 'stop', 'whether_test',
                     'cur_in_memory', 'nxt_in_memory', 'cur_vis', 'nxt_vis', 'target_cluster_has_disappeared_for_last_time_dummy']

    def __init__(self, x_var=None, y_var=None, bin_width=None, save_dir=None):
        self.x_var = x_var
        self.y_var = y_var
        self.bin_width = bin_width
        self.save_dir = save_dir
        self.sm_handler = gdh.smooths_handler()

    def _categorize_features(self, temporal_vars):
        self.temporal_vars = [
            x for x in temporal_vars if x in self.y_var.columns]
        self.spatial_vars = [
            x for x in self.y_var.columns if x not in temporal_vars]
        print('Spatial variables:', np.array(self.spatial_vars))

        self.temporal_sub = self.y_var[self.temporal_vars]
        self.spatial_sub_unscaled = self.y_var[self.spatial_vars]

    def streamline_pgam(self, temporal_vars=None, neural_cluster_number=10, num_total_trials=10):
        self.prepare_for_pgam(temporal_vars, num_total_trials)
        self._add_temporal_features_to_model(plot_each_feature=False)
        self._add_spatial_features_to_model(plot_each_feature=False)
        self.run_pgam(neural_cluster_number=neural_cluster_number)
        self.post_processing_results(neural_cluster_number=neural_cluster_number)
        self.save_results()

    def prepare_for_pgam(self, temporal_vars=None, num_total_trials=10):
        if temporal_vars is None:
            temporal_vars = self.temporal_vars
        self._categorize_features(temporal_vars)
        self._scale_features()
        self._get_mock_trials_df(num_total_trials)
        

    def run_pgam(self, neural_cluster_number=5):
        self.neural_cluster_number = neural_cluster_number
        link = sm.genmod.families.links.log()
        self.poissFam = sm.genmod.families.family.Poisson(link=link)
        self.spk_counts = self.x_var.iloc[:, neural_cluster_number].values
        self.cluster_name = self.x_var.columns[neural_cluster_number]

        # create the pgam model
        self.pgam = general_additive_model(self.sm_handler,
                                           self.sm_handler.smooths_var,  # list of covariate we want to include in the model
                                           self.spk_counts,  # vector of spike counts
                                           self.poissFam  # poisson family with exponential link from statsmodels.api
                                           )

        # with all covariate, remove according to stat testing, and then refit
        self.full, self.reduced = self.pgam.fit_full_and_reduced(self.sm_handler.smooths_var,
                                                                 th_pval=0.001,  # pval for significance of covariate icluseioon
                                                                 max_iter=10 ** 2,  # max number of iteration
                                                                 use_dgcv=True,  # learn the smoothing penalties by dgcv
                                                                 trial_num_vec=self.trial_ids,
                                                                 filter_trials=self.train_trials,
                                                                 )
        try:
            print('Minimal subset of variables driving the activity:')
            print(self.reduced.var_list)
            self.reduced_vars = self.reduced.var_list
        except Exception as e:
            print(f"Error occurred while printing reduced variable list: {e}")

    def post_processing_results(self, neural_cluster_number=None):
        # string with the neuron identifier
        if neural_cluster_number is None:
            neuron_id = 'neuron_000_session_1_monkey_001'
        else:
            neuron_id = f'neuron_{neural_cluster_number}'
        # dictionary containing some information about the neuron, keys must be strings and values can be anything since are stored with type object.
        info_save = {'x': 100,
                     'y': 801.2,
                     'brain_region': 'V1',
                     'subject': 'monkey_001'
                     }

        print('Post-processing results...')
        # assume that we used 90% of the trials for training, 10% for evaluation
        self.res = postprocess_results(neuron_id, self.spk_counts, self.full, self.reduced, self.train_trials, self.sm_handler, self.poissFam, self.trial_ids,
                                       var_zscore_par=None, info_save=info_save, bins=self.kernel_h_length)


    def plot_results(
        self,
        plot_vars_in_reduced_list_only=False,
        plot_var_order=None,
    ):

        self._rename_variables_in_results()

        res_vars = np.array(self.res['variable'])

        # ---------------------------------
        # Determine variable ordering
        # ---------------------------------
        if plot_var_order is not None:
            # keep only variables that exist
            ordered_vars = [v for v in plot_var_order if v in res_vars]

        else:
            # default: preserve order in res
            ordered_vars = list(res_vars)

        if plot_vars_in_reduced_list_only:
            ordered_vars = [
                v for v in ordered_vars
                if v in self.reduced_vars
            ]

        if len(ordered_vars) == 0:
            print('No variables to plot after filtering.')
            return

        # ---------------------------------
        # Map variables → indices
        # ---------------------------------
        indices_of_vars_to_plot = np.concatenate([
            np.where(res_vars == v)[0]
            for v in ordered_vars
        ])

        # ---------------------------------
        # Plot
        # ---------------------------------
        plot_modeling_result.plot_pgam_tuning_curvetions(
            self.res,
            indices_of_vars_to_plot=indices_of_vars_to_plot,
        )
        
        
    def load_pgam_results(self, neural_cluster_number):
        self.cluster_name = neural_cluster_number

        self.res, self.reduced_vars, self.meta, self.full_model_data = pgam_utils.load_full_results_npz(self.save_dir,
                                                                                                         self.cluster_name)
        
        # Reconstruct self.full with loaded attributes
        full_attributes = {
            'var_list': self.full_model_data.get('var_list'),
            'beta': self.full_model_data.get('beta'),
            'time_bin': self.full_model_data.get('time_bin'),
            'AIC': self.meta.get('full_AIC', np.nan),
        }
        self.full = LoadedModelData(**full_attributes)
        
        # Reconstruct self.reduced with loaded attributes
        reduced_attributes = {
            'var_list': self.reduced_vars,
            'AIC': self.meta.get('reduced_AIC', np.nan),
        }
        self.reduced = LoadedModelData(**reduced_attributes)

    def save_results(self, save_dir=None):
        # after you compute self.res = postprocess_results(...):
        extra_meta = {
            "bin_width": float(self.bin_width),
            "neuron_index": int(self.neural_cluster_number),
            "trial_count": int(len(np.unique(self.trial_ids))),
            "reduced_AIC": float(getattr(self.reduced, "AIC", np.nan)) if hasattr(self.reduced, "AIC") else np.nan,
            "full_AIC": float(getattr(self.full, "AIC", np.nan)) if hasattr(self.full, "AIC") else np.nan,
        }
        
        if save_dir is None:
            save_dir = self.save_dir
            
        # make sure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract full model data if available
        full_time_bin = getattr(self.full, "time_bin", None) if hasattr(self, "full") else None
        full_var_list = getattr(self.full, "var_list", None) if hasattr(self, "full") else None
        full_beta = getattr(self.full, "beta", None) if hasattr(self, "full") else None
        
        pgam_utils.save_full_results_npz(save_dir,
                                         self.cluster_name,
                                         self.res,                       # the structured array
                                         getattr(self.reduced, "var_list", []),
                                         extra_meta,
                                         full_time_bin=full_time_bin,
                                         full_var_list=full_var_list,
                                         full_beta=full_beta)

    def _scale_features(self):
        # since temporal variables are all dummy variables, we only need to scale the spatial variables
        scaler = StandardScaler()
        spatial_sub = scaler.fit_transform(self.spatial_sub_unscaled)
        self.spatial_sub = pd.DataFrame(
            spatial_sub, columns=self.spatial_sub_unscaled.columns)

    def _get_mock_trials_df(self, num_total_trials=10):
        self.num_total_trials = num_total_trials
        num_data_points = self.y_var.shape[0]
        num_repeats = math.ceil(num_data_points/num_total_trials)
        trial_ids = np.repeat(np.arange(num_total_trials), num_repeats)
        self.trial_ids = trial_ids[:num_data_points]
        # take out 2/3 of the trials for training
        self.train_trials = self.trial_ids % 3 != 1

    def _add_temporal_features_to_model(self,
                                        # Duration of the kernel h(t) in seconds
                                        kernel_time_window=10,
                                        num_internal_knots=8,  # Number of internal knots used to represent h
                                        # the order of the base spline, the number of coefficient in the polinomial (ord =4 is cubic spline)
                                        order=4,
                                        plot_each_feature=True,
                                        ):
        # Define the B-spline parameters

        # length in time points of the kernel
        self.kernel_h_length = int(kernel_time_window / self.bin_width)
        if self.kernel_h_length % 2 == 0:
            self.kernel_h_length += 1

        # Iterate over columns in the temporal subset
        for column in self.temporal_sub.columns:
            # Add the covariate & evaluate the convolution
            self.sm_handler.add_smooth(
                column,
                [self.temporal_sub[column].values],
                is_temporal_kernel=True,
                ord=order,
                knots_num=num_internal_knots,
                trial_idx=self.trial_ids,
                kernel_length=self.kernel_h_length,
                kernel_direction=0,
                time_bin=self.bin_width
            )

            if plot_each_feature:
                plot_modeling_result.plot_smoothed_temporal_feature(
                    self.temporal_sub, column, self.sm_handler, self.kernel_h_length)

    def _add_spatial_features_to_model(self,
                                       # the order of the base spline, the number of coefficient in the polinomial (ord =4 is cubic spline)
                                       order=4,
                                       plot_each_feature=True,
                                       ):

        # Add the 1D spatial variable
        for column in self.spatial_sub.columns:
            column_values = self.spatial_sub[column].values

            # Remove the variable from smooths_var and smooths_dict if it exists
            if column in self.sm_handler.smooths_var:
                self.sm_handler.smooths_var.remove(column)
                self.sm_handler.smooths_dict.pop(column)

            # Define internal knots and extend them for boundary conditions
            int_knots = np.linspace(min(column_values), max(column_values), 6)
            knots = np.hstack(([int_knots[0]] * (order - 1),
                              int_knots, [int_knots[-1]] * (order - 1)))

            # Add the smooth variable
            self.sm_handler.add_smooth(
                column, [column_values],
                knots=[knots],
                ord=order,
                is_temporal_kernel=False,
                trial_idx=self.trial_ids,
                is_cyclic=[False]
            )

            if plot_each_feature:
                plot_modeling_result.plot_smoothed_spatial_feature(
                    self.spatial_sub, column, self.sm_handler)


    def _rename_variables_in_results(self):
        variable = self.res['variable']
        # rename each variable to the corresponding label
        variable[variable == 'catching_ff'] = 'caught firefly (this bin)'
        variable[variable ==
                 'min_target_cluster_has_disappeared_for_last_time_dummy'] = 'target cluster disappeared (first time)'
        variable[variable ==
                 'max_target_cluster_visible_dummy'] = 'target cluster visible'
        variable[variable == 'gaze_world_y'] = 'gaze position (y)'
        variable[variable == 'speed'] = 'monkey speed'
        variable[variable == 'ang_speed'] = 'monkey acceleration'
        variable[variable == 'ang_accel'] = 'change in acceleration (jerk)'
        variable[variable == 'accel'] = 'change in angular accel'
        variable[variable == 'avg_target_cluster_last_seen_distance'] = 'distance to last-seen target cluster'
        variable[variable ==
                 'avg_target_cluster_last_seen_angle'] = 'angle to last-seen target cluster'
        variable[variable ==
                 'target_cluster_has_disappeared_for_last_time_dummy'] = 'target cluster disappeared (final time)'
        self.res['variable'] = variable


    def compute_variance_explained(self, use_train=True, filtwidth=2):
        """
        Reproduce MATLAB-style variance explained:
        - smooth spikes
        - smooth model prediction
        - compute 1 - SSE/SST
        """

        import numpy as np

        # --- Select train or eval ---
        if use_train:
            use_mask = self.train_trials
        else:
            use_mask = ~self.train_trials

        # --- Get spike counts ---
        spikes = self.spk_counts[use_mask]

        # --- Build design matrix ---
        exog, _ = self.sm_handler.get_exog_mat_fast(self.full.var_list)
        exog = exog[use_mask]

        # --- Model prediction (counts per bin) ---
        eta = exog @ self.full.beta
        mu = self.poissFam.fitted(eta)   # expected spike count per bin

        dt = self.full.time_bin

        # Convert to firing rate (Hz)
        fr_hat = mu / dt
        fr = spikes / dt

        # --- Gaussian smoothing (MATLAB style) ---
        t = np.linspace(-2*filtwidth, 2*filtwidth, 4*filtwidth + 1)
        h = np.exp(-t**2 / (2*filtwidth**2))
        h = h / np.sum(h)

        smooth_fr = np.convolve(fr, h, mode='same')
        smooth_fr_hat = np.convolve(fr_hat, h, mode='same')

        # --- Variance explained ---
        sse = np.sum((smooth_fr_hat - smooth_fr)**2)
        sst = np.sum((smooth_fr - np.mean(smooth_fr))**2)

        r2 = 1 - (sse / sst)

        return r2
        
    def _make_gaussian_kernel(self, filtwidth=2):
        import numpy as np
        t = np.linspace(-2 * filtwidth, 2 * filtwidth, 4 * filtwidth + 1)
        h = np.exp(-t**2 / (2 * filtwidth**2))
        h = h / np.sum(h)
        return h

    def _variance_explained_smoothed(self, counts, mu_counts, time_bin, filtwidth=2, use_mask=None):
        import numpy as np
        from scipy.signal import fftconvolve

        counts = np.asarray(counts)
        mu_counts = np.asarray(mu_counts)

        if use_mask is not None:
            counts = counts[use_mask]
            mu_counts = mu_counts[use_mask]

        # counts/bin -> Hz
        fr_obs = counts / time_bin
        fr_hat = mu_counts / time_bin

        h = self._make_gaussian_kernel(filtwidth=filtwidth)

        smooth_obs = fftconvolve(fr_obs, h, mode='same')
        smooth_hat = fftconvolve(fr_hat, h, mode='same')

        sse = np.nansum((smooth_hat - smooth_obs) ** 2)
        sst = np.nansum((smooth_obs - np.nanmean(smooth_obs)) ** 2)

        if sst == 0:
            return np.nan
        return 1.0 - (sse / sst)

    def _make_cv_train_mask_list(self, n_splits=5, random_state=0):
        """
        Returns a list of boolean masks (length = n_splits).
        Each mask is True for TRAIN bins and False for TEST bins.
        Split is done by trial_id to avoid leakage across time bins.
        """
        import numpy as np
        from sklearn.model_selection import KFold

        trial_ids = np.asarray(self.trial_ids)
        unique_trials = np.unique(trial_ids)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        train_mask_list = []
        for train_trial_idx, test_trial_idx in kf.split(unique_trials):
            train_trials = unique_trials[train_trial_idx]
            train_mask = np.isin(trial_ids, train_trials)
            train_mask_list.append(train_mask)

        return train_mask_list


    def _get_cv_filename(self, neural_cluster_number, n_splits, filtwidth):
        import os
        fname = f'cv_var_explained/neuron_{neural_cluster_number}_folds_{n_splits}_fw_{filtwidth}.npz'
        filename = os.path.join(self.save_dir, fname)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return filename
    
    def _save_cv_results(self, filename, result_dict):
        import numpy as np
        np.savez_compressed(
            filename,
            fold_r2_eval=result_dict['fold_r2_eval'],
            mean_r2_eval=result_dict['mean_r2_eval'],
            std_r2_eval=result_dict['std_r2_eval']
        )
        print(f'Saved CV results for neuron {self.cluster_name} at {filename}')
        
    def _load_cv_results(self, filename):
        import numpy as np
        data = np.load(filename, allow_pickle=True)
        return {
            'fold_r2_eval': data['fold_r2_eval'],
            'mean_r2_eval': float(data['mean_r2_eval']),
            'std_r2_eval': float(data['std_r2_eval'])
        }
                
    def _run_pgam_cv_compute_only(self,
                                neural_cluster_number=5,
                                n_splits=5,
                                filtwidth=2,
                                random_state=0):

        import numpy as np
        import statsmodels.api as sm

        self.neural_cluster_number = neural_cluster_number

        link = sm.genmod.families.links.log()
        self.poissFam = sm.genmod.families.family.Poisson(link=link)

        self.spk_counts = self.x_var.iloc[:, neural_cluster_number].values
        self.cluster_name = self.x_var.columns[neural_cluster_number]

        self.pgam = general_additive_model(
            self.sm_handler,
            self.sm_handler.smooths_var,
            self.spk_counts,
            self.poissFam
        )

        train_mask_list = self._make_cv_train_mask_list(
            n_splits=n_splits,
            random_state=random_state
        )

        fold_r2_eval = []

        for train_mask in train_mask_list:

            full_fit, _ = self.pgam.fit_full_and_reduced(
                self.sm_handler.smooths_var,
                th_pval=0.001,
                max_iter=10 ** 2,
                use_dgcv=True,
                trial_num_vec=self.trial_ids,
                filter_trials=train_mask
            )

            exog_full, _ = self.sm_handler.get_exog_mat_fast(full_fit.var_list)
            eta = exog_full @ full_fit.beta
            mu_counts = self.poissFam.fitted(eta)

            test_mask = ~train_mask

            r2_eval = self._variance_explained_smoothed(
                counts=self.spk_counts,
                mu_counts=mu_counts,
                time_bin=full_fit.time_bin,
                filtwidth=filtwidth,
                use_mask=test_mask
            )

            fold_r2_eval.append(r2_eval)

        fold_r2_eval = np.asarray(fold_r2_eval)

        return {
            'neuron': self.cluster_name,
            'neural_cluster_number': neural_cluster_number,
            'fold_r2_eval': fold_r2_eval,
            'mean_r2_eval': np.nanmean(fold_r2_eval),
            'std_r2_eval': np.nanstd(fold_r2_eval)
        }
            
    def run_pgam_cv(self,
                    neural_cluster_number=5,
                    n_splits=5,
                    filtwidth=2,
                    random_state=0,
                    force_recompute=False,
                    load_only=False,
                    ):

        import os

        filename = self._get_cv_filename(
            neural_cluster_number,
            n_splits,
            filtwidth
        )

        # Load if exists
        if not force_recompute:
            if os.path.exists(filename):
                print(f'Loading cached CV results for neuron {neural_cluster_number} at {filename}')
                return self._load_cv_results(filename)
            else:
                print(f'No cached CV results found for neuron {neural_cluster_number} at {filename}, computing...')
                if load_only:
                    raise FileNotFoundError(f'No cached CV results found for neuron {neural_cluster_number} at {filename}')

        # Compute
        out = self._run_pgam_cv_compute_only(
            neural_cluster_number=neural_cluster_number,
            n_splits=n_splits,
            filtwidth=filtwidth,
            random_state=random_state
        )

        # Save
        self._save_cv_results(filename, out)

        return out