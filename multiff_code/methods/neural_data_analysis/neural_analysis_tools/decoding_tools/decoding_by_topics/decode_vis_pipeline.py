import os
import pickle
from pathlib import Path

# Third-party imports
import pandas as pd
import statsmodels.api as sm

# Neuroscience specific imports
from neural_data_analysis.design_kits.design_around_event import event_binning
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import (
    pn_decoding_model_specs
)
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    collect_stop_data,
    prepare_stop_design
)
from neural_data_analysis.topic_based_neural_analysis.ff_visibility import decode_vis_utils
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event

# Additional imports

# Machine Learning imports


class FFVisDecodingRunner:
    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        t_max=0.20,
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width
        self.t_max = t_max

        # will be filled during setup
        self.datasets = None
        self.comparisons = None
        
        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)

        

    def _collect_data(self, exists_ok=True):
        """
        Collect and prepare data for decoding.
        
        Args:
            exists_ok: If True, load cached design matrices if they exist.
        """
        # Try to load cached design matrices
        if exists_ok and self._load_design_matrices():
            print('[_collect_data] Using cached design matrices')
        else:
            self.pn, self.datasets, self.comparisons = (
                collect_stop_data.collect_stop_data_func(
                    self.raw_data_folder_path
                )
            )
            self.pn.make_or_retrieve_ff_dataframe()

            print('[_collect_data] Computing design matrices from scratch')
            (
                self.spike_data_w_history,
                self.binned_feats_sc,
                self.meta_used,
            ) = self._prepare_design_matrices()
            
            # Save the computed design matrices for future use
            self._save_design_matrices()



    def _prepare_design_matrices(self):
        new_seg_info, events_with_stats = decode_vis_utils.prepare_new_seg_info(
            self.pn.ff_dataframe,
            self.pn.bin_width,
        )

        (
            binned_spikes,
            binned_feats,
            offset_log,
            meta_used,
            meta_groups,
        ) = prepare_stop_design.build_stop_design(
            new_seg_info,
            events_with_stats,
            self.pn.monkey_information,
            self.pn.spikes_df,
            self.pn.ff_dataframe,
            datasets=self.datasets,
            bin_dt=self.pn.bin_width,
            add_ff_visible_info=True,
            add_retries_info=False,
        )

        if 'global_burst_id' not in meta_used.columns:
            meta_used = meta_used.merge(
                new_seg_info[['event_id', 'global_burst_id']],
                on='event_id',
                how='left',
            )

        binned_feats_sc, scaled_cols = event_binning.selective_zscore(
            binned_feats
        )
        binned_feats_sc = sm.add_constant(
            binned_feats_sc,
            has_constant='add',
        )

        bin_df = spike_history.make_bin_df_from_stop_meta(meta_used)

        (
            spike_data_w_history,
            basis,
            colnames,
            meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.pn.spikes_df,
            bin_df=bin_df,
            X_pruned=binned_spikes,
            meta_groups={},
            dt=self.bin_width,
            t_max=self.t_max,
        )

        return spike_data_w_history, binned_feats_sc, meta_used

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'ff_vis_decoding',
        )
    
    def _get_design_matrix_paths(self):
        """Get file paths for cached design matrices."""
        save_dir = Path(self._get_save_dir())
        return {
            'spike_data_w_history': save_dir / 'spike_data_w_history.pkl',
            'binned_feats_sc': save_dir / 'binned_feats_sc.pkl',
            'meta_used': save_dir / 'meta_used.pkl',
        }
    
    def _save_design_matrices(self):
        """Save design matrices to disk."""
        paths = self._get_design_matrix_paths()
        save_dir = Path(self._get_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        
        data_to_save = {
            'spike_data_w_history': self.spike_data_w_history,
            'binned_feats_sc': self.binned_feats_sc,
            'meta_used': self.meta_used,
        }
        
        for key, data in data_to_save.items():
            try:
                with open(paths[key], 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'[_save_design_matrices] Saved {key} to {paths[key]}')
            except Exception as e:
                print(f'[_save_design_matrices] WARNING: could not save {key}: {type(e).__name__}: {e}')
    
    
    def _load_design_matrices(self):
        """Load design matrices from disk. Returns True if successful, False otherwise."""
        paths = self._get_design_matrix_paths()
        
        # Check if all files exist
        if not all(path.exists() for path in paths.values()):
            return False
        
        try:
            with open(paths['spike_data_w_history'], 'rb') as f:
                self.spike_data_w_history = pickle.load(f)
            print(f'[_load_design_matrices] Loaded spike_data_w_history from {paths["spike_data_w_history"]}')
            
            with open(paths['binned_feats_sc'], 'rb') as f:
                self.binned_feats_sc = pickle.load(f)
            print(f'[_load_design_matrices] Loaded binned_feats_sc from {paths["binned_feats_sc"]}')
            
            with open(paths['meta_used'], 'rb') as f:
                self.meta_used = pickle.load(f)
            print(f'[_load_design_matrices] Loaded meta_used from {paths["meta_used"]}')
            
            return True
        except Exception as e:
            print(f'[_load_design_matrices] WARNING: could not load design matrices: {type(e).__name__}: {e}')
            return False

    def run(self, n_splits=5, save_dir=None, design_matrices_exists_ok=True):
        """
        Run the FF visibility decoding pipeline.
        
        Args:
            n_splits: Number of cross-validation splits.
            save_dir: Directory to save results. If None, uses default.
            exists_ok: If True, load cached design matrices if they exist.
        """
        self._collect_data(exists_ok=design_matrices_exists_ok)

        if save_dir is None:
            self.save_dir = self._get_save_dir()
        else:
            self.save_dir = Path(save_dir)

        all_results = []

        for model_name, spec in pn_decoding_model_specs.MODEL_SPECS.items():
            config = cv_decoding.DecodingRunConfig(
                # regression
                regression_model_class=spec.get(
                    'regression_model_class', None
                ),
                regression_model_kwargs=spec.get(
                    'regression_model_kwargs', {}
                ),
                # classification
                classification_model_class=spec.get(
                    'classification_model_class', None
                ),
                classification_model_kwargs=spec.get(
                    'classification_model_kwargs', {}
                ),
                # shared
                use_early_stopping=False,
            )


            print('model_name:', model_name)
            print('config:', config)

            results_df = cv_decoding.run_cv_decoding(
                X=self.spike_data_w_history,
                y_df=self.binned_feats_sc,
                behav_features=None,
                groups=self.meta_used['event_id'].values,
                n_splits=n_splits,
                config=config,
                context_label='pooled',
                save_dir=self.save_dir,
            )

            results_df['model_name'] = model_name
            all_results.append(results_df)

        all_results_df = pd.concat(all_results, ignore_index=True)
        return all_results_df
