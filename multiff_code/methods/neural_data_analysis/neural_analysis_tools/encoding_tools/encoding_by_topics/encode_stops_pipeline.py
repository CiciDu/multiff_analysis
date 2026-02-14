import os
import pickle
from pathlib import Path

# Third-party imports
import pandas as pd

# Neuroscience-specific imports
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding
)
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    assemble_stop_design
)
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import (
    pn_decoding_model_specs
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event


class StopEncodingRunner:
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
        self.stop_meta_used = None
        self.spike_data_w_history = None
        self.stop_binned_feats = None

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, n_splits=5, save_dir=None, design_matrices_exists_ok=True, model_specs=None, shuffle_mode='none'):
        """
        Run stop-event encoding.
        """
        self.model_specs = model_specs if model_specs is not None else pn_decoding_model_specs.MODEL_SPECS
        self._collect_data(exists_ok=design_matrices_exists_ok)

        if save_dir is None:
            save_dir = self._get_save_dir()
        if shuffle_mode != 'none':
            save_dir = Path(save_dir) / f'shuffle_{shuffle_mode}'
            save_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        for model_name, spec in self.model_specs.items():
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
                y_df=self.stop_binned_feats,
                behav_features=None,
                groups=self.stop_meta_used['event_id'].values,
                n_splits=n_splits,
                config=config,
                context_label='pooled',
                save_dir=save_dir,
                model_name=model_name,
                shuffle_mode=shuffle_mode,
            )

            results_df['model_name'] = model_name
            all_results.append(results_df)

        return pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_data(self, exists_ok=True):
        """
        Collect and prepare data for stop decoding.
        """
        if exists_ok and self._load_design_matrices():
            print('[StopEncodingRunner] Using cached design matrices')
            return

        print('[StopEncodingRunner] Computing design matrices from scratch')

        (
            self.pn,
            stop_binned_spikes,
            self.stop_binned_feats,
            offset_log,
            self.stop_meta_used,
            stop_meta_groups,
        ) = assemble_stop_design.assemble_stop_design_func(
            self.raw_data_folder_path,
            self.bin_width,
        )

        bin_df = spike_history.make_bin_df_from_stop_meta(self.stop_meta_used)

        (
            self.spike_data_w_history,
            basis,
            colnames,
            meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.pn.spikes_df,
            bin_df=bin_df,
            X_pruned=stop_binned_spikes,
            meta_groups={},
            dt=self.bin_width,
            t_max=self.t_max,
        )

        self._save_design_matrices()

    # ------------------------------------------------------------------
    # Caching utilities
    # ------------------------------------------------------------------
    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'encoding_outputs/stop_encoder_outputs',
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            'spike_data_w_history': save_dir / 'spike_data_w_history.pkl',
            'stop_binned_feats': save_dir / 'stop_binned_feats.pkl',
            'stop_meta_used': save_dir / 'stop_meta_used.pkl',
        }

    def _save_design_matrices(self):
        save_dir = Path(self._get_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = self._get_design_matrix_paths()

        data_to_save = {
            'spike_data_w_history': self.spike_data_w_history,
            'stop_binned_feats': self.stop_binned_feats,
            'stop_meta_used': self.stop_meta_used,
        }

        for key, data in data_to_save.items():
            try:
                with open(paths[key], 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'[StopEncodingRunner] Saved {key} → {paths[key]}')
            except Exception as e:
                print(
                    f'[StopEncodingRunner] WARNING: could not save {key}: '
                    f'{type(e).__name__}: {e}'
                )

    def _load_design_matrices(self):
        paths = self._get_design_matrix_paths()

        if not all(p.exists() for p in paths.values()):
            return False

        try:
            with open(paths['spike_data_w_history'], 'rb') as f:
                self.spike_data_w_history = pickle.load(f)

            with open(paths['stop_binned_feats'], 'rb') as f:
                self.stop_binned_feats = pickle.load(f)

            with open(paths['stop_meta_used'], 'rb') as f:
                self.stop_meta_used = pickle.load(f)

            print('[StopEncodingRunner] Loaded cached design matrices')
            return True

        except Exception as e:
            print(
                f'[StopEncodingRunner] WARNING: could not load design matrices: '
                f'{type(e).__name__}: {e}'
            )
            return False
