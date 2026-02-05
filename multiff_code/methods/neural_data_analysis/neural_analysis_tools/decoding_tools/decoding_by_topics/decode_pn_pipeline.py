import os
import pickle
from pathlib import Path

# Third-party imports
import pandas as pd

# Decoding
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding
)

# PN-specific imports
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import (
    pn_aligned_by_event
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import (
    pn_decoding_model_specs
)
from neural_data_analysis.design_kits.design_by_segment import (
    spike_history,
    temporal_feats,
    create_pn_design_df,
)


class PNDecodingRunner:
    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        t_max=0.20,
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width
        self.t_max = t_max

        # Filled during setup
        self.design_df = None
        self.trial_ids = None
        self.spike_data_w_history = None
        
        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(raw_data_folder_path=self.raw_data_folder_path, bin_width=self.bin_width)
    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def _collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            print('[PNDecodingRunner] Using cached design matrices')
            return

        print('[PNDecodingRunner] Computing design matrices from scratch')

        pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

        pn.prep_data_to_analyze_planning(
            planning_data_by_point_exists_ok=True
        )

        pn.rebin_data_in_new_segments(
            cur_or_nxt='cur',
            first_or_last='first',
            time_limit_to_count_sighting=2,
            start_t_rel_event=0,
            end_t_rel_event=1.5,
            rebinned_max_x_lag_number=2,
        )

        data = pn.rebinned_y_var.copy()
        trial_ids = data['new_segment']
        dt = pn.bin_width

        data = temporal_feats.add_stop_and_capture_columns(
            data,
            trial_ids,
            pn.ff_caught_T_new,
        )

        design_df, meta0, meta = (
            create_pn_design_df.get_initial_design_df(
                data,
                dt,
                trial_ids,
            )
        )

        cluster_cols = [
            c for c in pn.rebinned_x_var.columns
            if c.startswith('cluster_')
        ]

        df_Y = pn.rebinned_x_var[cluster_cols]
        df_Y.columns = (
            df_Y.columns
            .str.replace('cluster_', '')
            .astype(int)
        )

        bin_df = create_pn_design_df.make_bin_df_for_pn(
            pn.rebinned_x_var,
            pn.bin_edges,
        )

        (
            spike_data_w_history,
            basis,
            colnames,
            meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=pn.spikes_df,
            bin_df=bin_df,
            X_pruned=df_Y,
            meta_groups={},
            dt=self.bin_width,
            t_max=self.t_max,
        )

        self.pn = pn
        self.design_df = design_df
        self.trial_ids = trial_ids
        self.spike_data_w_history = spike_data_w_history

        self._save_design_matrices()

    # ------------------------------------------------------------------
    # Caching utilities
    # ------------------------------------------------------------------
    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'pn_decoder_outputs',
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            'spike_data_w_history': save_dir / 'spike_data_w_history.pkl',
            'design_df': save_dir / 'design_df.pkl',
            'trial_ids': save_dir / 'trial_ids.pkl',
        }

    def _save_design_matrices(self):
        save_dir = Path(self._get_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = self._get_design_matrix_paths()

        data_to_save = {
            'spike_data_w_history': self.spike_data_w_history,
            'design_df': self.design_df,
            'trial_ids': self.trial_ids,
        }

        for key, data in data_to_save.items():
            try:
                with open(paths[key], 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'[PNDecodingRunner] Saved {key} → {paths[key]}')
            except Exception as e:
                print(
                    f'[PNDecodingRunner] WARNING: could not save {key}: '
                    f'{type(e).__name__}: {e}'
                )

    def _load_design_matrices(self):
        paths = self._get_design_matrix_paths()

        if not all(p.exists() for p in paths.values()):
            return False

        try:
            with open(paths['spike_data_w_history'], 'rb') as f:
                self.spike_data_w_history = pickle.load(f)

            with open(paths['design_df'], 'rb') as f:
                self.design_df = pickle.load(f)

            with open(paths['trial_ids'], 'rb') as f:
                self.trial_ids = pickle.load(f)

            print('[PNDecodingRunner] Loaded cached design matrices')
            return True

        except Exception as e:
            print(
                f'[PNDecodingRunner] WARNING: could not load design matrices: '
                f'{type(e).__name__}: {e}'
            )
            return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, n_splits=5, save_dir=None, design_matrices_exists_ok=True):
        self._collect_data(exists_ok=design_matrices_exists_ok)

        if save_dir is None:
            save_dir = self._get_save_dir()

        all_results = []

        for model_name, spec in pn_decoding_model_specs.MODEL_SPECS.items():
            config = cv_decoding.DecodingRunConfig(
                regression_model_class=spec.get(
                    'regression_model_class', None
                ),
                regression_model_kwargs=spec.get(
                    'regression_model_kwargs', {}
                ),
                classification_model_class=spec.get(
                    'classification_model_class', None
                ),
                classification_model_kwargs=spec.get(
                    'classification_model_kwargs', {}
                ),
                use_early_stopping=False,
            )

            print('model_name:', model_name)
            print('config:', config)

            results_df = cv_decoding.run_cv_decoding(
                X=self.spike_data_w_history,
                y_df=self.design_df,
                behav_features=None,
                groups=self.trial_ids,
                n_splits=n_splits,
                config=config,
                context_label='pooled',
                save_dir=save_dir,
            )

            results_df['model_name'] = model_name
            all_results.append(results_df)

        return pd.concat(all_results, ignore_index=True)
