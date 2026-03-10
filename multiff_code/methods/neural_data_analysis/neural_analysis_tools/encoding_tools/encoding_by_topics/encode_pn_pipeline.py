import os

# Third-party imports

# self.pn-specific imports
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import (
    pn_aligned_by_event
)
from neural_data_analysis.design_kits.design_by_segment import (
    temporal_feats,
    create_pn_design_df,
)
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encoder_gam_helper,
    encode_pn_utils
)

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics.base_encoding_runner import (
    BaseEncodingRunner,
)


from neural_data_analysis.design_kits.design_by_segment import spatial_feats

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encoding_design_utils


class PNEncodingRunner(BaseEncodingRunner):
    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        encoder_prs=None,
    ):
        super().__init__(raw_data_folder_path, bin_width=bin_width, encoder_prs=encoder_prs)

        self.var_categories = encoder_gam_helper.PN_VAR_CATEGORIES
        
        print('var_categories:', self.var_categories)

        self.lam_grid = {
            "lam_f": [50, 100, 200],
            "lam_g": [50, 100, 200],
            "lam_h": [5, 10, 30],
            "lam_p": [10],
        }

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _prepare_to_collect_data(self):
        print('[PNEncodingRunner] Computing design matrices from scratch')

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

        self.pn.prep_data_to_analyze_planning(
            planning_data_by_point_exists_ok=True
        )

        self.pn.rebin_data_in_new_segments(
            cur_or_nxt='cur',
            first_or_last='first',
            time_limit_to_count_sighting=2,
            start_t_rel_event=0,
            end_t_rel_event=1.5,
            rebinned_max_x_lag_number=2,
        )

        self.pn.global_bins_2d = self.pn.bin_edges

        return

    def _make_binned_feats(self):

        rebinned_y_var = self.pn.rebinned_y_var.copy()
        trial_ids = rebinned_y_var['new_segment']

        rebinned_y_var = temporal_feats.add_stop_and_capture_columns(
            rebinned_y_var,
            trial_ids,
            self.pn.ff_caught_T_new,
        )

        # add cur_vis_on, cur_vis_off, nxt_vis_on, nxt_vis_off
        rebinned_y_var = spatial_feats.add_transition_columns(
            rebinned_y_var, rebinned_y_var['new_segment'].values,
            stems=('cur_vis', 'nxt_vis'), inplace=False
        )

        self.rebinned_y_var = rebinned_y_var.copy()

        # maybe this will be useful in the future
        # init_binned_feats, meta0, meta = (
        #     create_pn_design_df.get_pn_design_base(
        #         rebinned_y_var,
        #         self.pn.bin_width,
        #         trial_ids,
        #     )
        # )

        cluster_cols = [
            c for c in self.pn.rebinned_x_var.columns if c.startswith('cluster_')]
        self.binned_spikes = self.pn.rebinned_x_var[cluster_cols]
        self.binned_spikes.columns = (
            self.binned_spikes.columns
            .str.replace('cluster_', '')
            .astype(int)
        )

        design_kwargs = self._encoding_design_kwargs()

        linear_vars = encoding_design_utils.DEFAULT_TUNING_VARS_NO_WRAP + \
            ['time_since_target_last_seen', 'time_since_last_capture',
             'cur_ff_distance', 'nxt_ff_distance',
             'cur_ff_angle', 'nxt_ff_angle']

        (self.binned_feats,
         self.temporal_meta,
         self.tuning_meta,
         ) = encode_pn_utils.build_pn_encoding_design(
            rebinned_y_var,
            self.pn.monkey_information,
            global_bins_2d=self.pn.global_bins_2d,
            bin_width=self.pn.bin_width,
            linear_vars=linear_vars,
            **design_kwargs,
        )

    def collect_data(self, exists_ok=True):
        if exists_ok and self._load_design_matrices():
            print('[PNEncodingRunner] Using cached design matrices')
            return

        self._prepare_to_collect_data()

        self._make_binned_feats()

        data = self.pn.rebinned_y_var.copy()

        cluster_cols = [
            c for c in self.pn.rebinned_x_var.columns
            if c.startswith('cluster_')
        ]
        self.binned_spikes = self.pn.rebinned_x_var[cluster_cols]
        self.binned_spikes.columns = (
            self.binned_spikes.columns
            .str.replace('cluster_', '')
            .astype(int)
        )
        self.binned_spikes = self.binned_spikes.reset_index(drop=True)

        self.trial_ids = data['new_segment']

        self.bin_df = create_pn_design_df.make_bin_df_for_pn(
            self.pn.rebinned_x_var,
            self.pn.bin_edges,
        )
        # Build encoding design (behavioral + spike history) for GAM modeling
        self.binned_feats = self.binned_feats.reset_index(drop=True)

        self._prepare_spike_history_components()

        self._make_structured_meta_groups()

        self._save_design_matrices()

    # def _prepare_spike_history_components(self):
    #     """Build encoding spike history if not yet available (e.g. after cache load)."""
    #     if self.X_hist is not None:
    #         return
    #     if self.binned_feats is None or self.binned_spikes is None:
    #         raise RuntimeError("Run collect_data first.")
    #     self.collect_data(exists_ok=False)

    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'encoding_outputs/pn_encoder_outputs',
        )

    def get_gam_results_subdir(self) -> str:
        return "pn_gam_results"
