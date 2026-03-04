import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import decode_stops_design
from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import encode_stops_utils, encoding_design_utils

# Third-party imports
import numpy as np
import pandas as pd


from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import multiff_encoding_params
from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_fit,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam.one_ff_gam_fit import (
    FitResult,
    GroupSpec,
)


from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_helpers import (
    encode_stops_gam_helper,
)

from neural_data_analysis.neural_analysis_tools.encoding_tools.encoding_by_topics.base_encoding_runner import (
    BaseEncodingRunner,
)


class StopEncodingRunner(BaseEncodingRunner):
    def __init__(
        self,
        raw_data_folder_path,
        bin_width=0.04,
        encoder_prs=None,
    ):
        super().__init__(raw_data_folder_path, 
                         bin_width=bin_width,
                         encoder_prs=encoder_prs)

        self.var_categories = encode_stops_gam_helper.STOP_VAR_CATEGORIES



    def collect_data(self, exists_ok=True,     
                      tuning_feature_mode='boxcar_only' # can be 'raw_only', 'boxcar_only', 'raw_plus_boxcar'
                      ):
        """
        Collect and prepare data for stop encoding.
        """
        if exists_ok and self._load_design_matrices():
            print('[StopEncodingRunner] Using cached design matrices')
            return
        self.pn, datasets, new_seg_info, events_with_stats = (
            decode_stops_design._prepare_stop_design_inputs(
                self.raw_data_folder_path,
                self.bin_width,
            )
        )

        print('[EncodingRunner] Computing design matrices from scratch')
        design_kwargs = self._encoding_design_kwargs()
        design_kwargs['tuning_feature_mode'] = tuning_feature_mode

        (
            self.pn,
            self.binned_spikes,
            self.binned_feats,
            self.meta_df_used,
            self.temporal_meta,
            self.tuning_meta,
        ) = encode_stops_utils.build_stop_encoding_design(
            self.pn,
            datasets,
            new_seg_info,
            events_with_stats,
            self.bin_width,
            **design_kwargs,
        )

        self.bin_df = spike_history.make_bin_df_from_meta_df(
            self.meta_df_used)

        self._prepare_spike_history_components()
        self._make_structured_meta_groups()
        self._save_design_matrices()

    # ------------------------------------------------------------------
    # Caching utilities
    # ------------------------------------------------------------------
    def _get_save_dir(self):
        return os.path.join(
            self.pn.planning_and_neural_folder_path,
            'encoding_outputs/stop_encoder_outputs',
        )

    def get_gam_results_subdir(self) -> str:
        return "stop_gam_results"

