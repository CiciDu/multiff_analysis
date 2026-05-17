"""Behavioral-feature allowlists per decoding task (shared by design matrices and saved results)."""

from __future__ import annotations

from typing import Any, FrozenSet, Optional, Sequence

import pandas as pd

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_feature_selection.fs_select_feats import (
    keep as FS_FEATS_KEEP,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_feature_selection.pn_select_feats import (
    keep as PN_FEATS_KEEP,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_feature_selection.stops_select_feats import (
    keep as STOPS_FEATS_KEEP,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_feature_selection.vis_select_feats import (
    keep as VIS_FEATS_KEEP,
)

_FEATS_KEEP_BY_TASK_CLASS_NAME: dict[str, Sequence[str]] = {
    "PNTask": PN_FEATS_KEEP,
    "FSTask": FS_FEATS_KEEP,
    "VisTask": VIS_FEATS_KEEP,
    "StopTask": STOPS_FEATS_KEEP,
}


def feats_keep_frozenset_for_task_class(task_class: Any) -> Optional[FrozenSet[str]]:
    """Return allowed behavioral feature names for this task class, or None if unknown."""
    for cls in getattr(task_class, "__mro__", ()):
        seq = _FEATS_KEEP_BY_TASK_CLASS_NAME.get(cls.__name__)
        if seq is not None:
            return frozenset(seq)
    return None


def filter_decoding_results_by_task_feat_keep(
    results_df: pd.DataFrame,
    task_class: Any,
) -> pd.DataFrame:
    """Keep only rows whose ``behav_feature`` is in the task's feature allowlist."""
    keep = feats_keep_frozenset_for_task_class(task_class)
    if keep is None or results_df is None or len(results_df) == 0:
        return results_df
    if "behav_feature" not in results_df.columns:
        return results_df
    return results_df.loc[results_df["behav_feature"].isin(keep)].reset_index(drop=True)
