from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from neural_data_analysis.design_kits.design_by_segment import spike_history
from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    decode_stops_utils,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding,
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import (
    pn_decoding_model_specs,
)
from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    decode_stops_design,
)


DEFAULT_STOP_CANONCORR_VARS = [
    "speed",
    "ang_speed",
    "accel",
    "time_rel_to_event_start",
    "cluster_progress_c",
]


class StopDecodingRunner:
    """
    Stop decoding runner. CV model-spec decoding via run(); one-FF style
    population decoding (CCA + linear readout) via run_one_ff_style().
    """

    def __init__(
        self,
        raw_data_folder_path,
        bin_width: float = 0.04,
        t_max: float = 0.20,
        use_spike_history_for_one_ff: bool = False,
    ):
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width
        self.t_max = t_max
        self.use_spike_history_for_one_ff = use_spike_history_for_one_ff

        self.stop_meta_used = None
        self.stop_feats_to_decode = None
        self.stop_binned_spikes = None
        self.spike_data_w_history = None
        self.stats: Dict = {}

        self.pn = pn_aligned_by_event.PlanningAndNeuralEventAligned(
            raw_data_folder_path=self.raw_data_folder_path,
            bin_width=self.bin_width,
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        n_splits: int = 5,
        save_dir=None,
        design_matrices_exists_ok: bool = True,
        model_specs=None,
        shuffle_mode: str = "none",
    ):
        return self.run_cv_decoding(
            n_splits=n_splits,
            save_dir=save_dir,
            design_matrices_exists_ok=design_matrices_exists_ok,
            model_specs=model_specs,
            shuffle_mode=shuffle_mode,
        )

    def run_one_ff_style(
        self,
        *,
        design_matrices_exists_ok: bool = True,
        save_dir=None,
        canoncorr_varnames: Optional[Sequence[str]] = None,
        readout_varnames: Optional[Sequence[str]] = None,
        readout_n_splits: int = 5,
        readout_cv_mode: str = "blocked_time_buffered",
        readout_buffer_samples: int = 20,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        self._collect_data(exists_ok=design_matrices_exists_ok)
        if save_dir is None:
            save_dir = Path(self._get_save_dir()) / "one_ff_style"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        canoncorr = self.compute_canoncorr(
            varnames=canoncorr_varnames,
            save_path=str(save_dir / "canoncorr.pkl"),
            load_if_exists=load_if_exists,
            verbose=verbose,
        )
        readout = self.regress_popreadout(
            varnames=readout_varnames,
            n_splits=readout_n_splits,
            cv_mode=readout_cv_mode,
            buffer_samples=readout_buffer_samples,
            save_path=str(save_dir / "lineardecoder.pkl"),
            load_if_exists=load_if_exists,
            verbose=verbose,
        )
        return {"canoncorr": canoncorr, "readout": readout, "stats": self.stats}

    def run_cv_decoding(
        self,
        *,
        n_splits: int = 5,
        save_dir=None,
        design_matrices_exists_ok: bool = True,
        model_specs=None,
        shuffle_mode: str = "none",
    ) -> pd.DataFrame:
        self.model_specs = (
            model_specs if model_specs is not None else pn_decoding_model_specs.MODEL_SPECS
        )
        self._collect_data(exists_ok=design_matrices_exists_ok)

        if save_dir is None:
            save_dir = self._get_save_dir()
        if shuffle_mode != "none":
            save_dir = Path(save_dir) / f"shuffle_{shuffle_mode}"
            save_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        for model_name, spec in self.model_specs.items():
            config = cv_decoding.DecodingRunConfig(
                regression_model_class=spec.get("regression_model_class", None),
                regression_model_kwargs=spec.get("regression_model_kwargs", {}),
                classification_model_class=spec.get("classification_model_class", None),
                classification_model_kwargs=spec.get("classification_model_kwargs", {}),
                use_early_stopping=False,
            )
            print("[cv] model_name:", model_name)
            results_df = cv_decoding.run_cv_decoding(
                X=self.spike_data_w_history,
                y_df=self.stop_feats_to_decode,
                behav_features=None,
                groups=self.stop_meta_used["event_id"].values,
                n_splits=n_splits,
                config=config,
                context_label="pooled",
                save_dir=save_dir,
                model_name=model_name,
                shuffle_mode=shuffle_mode,
            )
            results_df["model_name"] = model_name
            all_results.append(results_df)

        return pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # One-FF-style decoding methods
    # ------------------------------------------------------------------
    def _get_numeric_target_df(self) -> pd.DataFrame:
        y_df = self.stop_feats_to_decode.copy()
        keep_cols = []
        for c in y_df.columns:
            if c == "const":
                continue
            if pd.api.types.is_numeric_dtype(y_df[c]) or pd.api.types.is_bool_dtype(y_df[c]):
                keep_cols.append(c)
        return y_df[keep_cols].astype(float)

    def _default_canoncorr_varnames(self) -> List[str]:
        y_df = self._get_numeric_target_df()
        vars_found = [v for v in DEFAULT_STOP_CANONCORR_VARS if v in y_df.columns]
        return vars_found if len(vars_found) > 0 else list(y_df.columns[: min(6, y_df.shape[1])])

    def _default_readout_varnames(self) -> List[str]:
        return list(self._get_numeric_target_df().columns)

    def _get_neural_matrix(self, use_spike_history: Optional[bool] = None) -> np.ndarray:
        if use_spike_history is None:
            use_spike_history = self.use_spike_history_for_one_ff
        if use_spike_history:
            return np.asarray(self.spike_data_w_history, dtype=float)
        return np.asarray(self.stop_binned_spikes, dtype=float)

    @staticmethod
    def _path_with_cv_params(
        save_path: Optional[str],
        *,
        cv_mode: str,
        n_splits: int,
        buffer_samples: int,
    ) -> Optional[Path]:
        """Derive save path that encodes cv config so different configs don't collide."""
        if save_path is None:
            return None
        p = Path(save_path)
        stem, suf = p.stem, p.suffix
        return p.parent / f"{stem}_cv{cv_mode}_n{n_splits}_buf{buffer_samples}{suf}"

    @staticmethod
    def _maybe_load_result(
        save_path: Optional[str],
        load_if_exists: bool,
        label: str,
        verbose: bool,
        *,
        cv_mode: Optional[str] = None,
        n_splits: Optional[int] = None,
        buffer_samples: Optional[int] = None,
    ):
        if (not load_if_exists) or save_path is None:
            return None
        if cv_mode is not None and n_splits is not None and buffer_samples is not None:
            p = StopDecodingRunner._path_with_cv_params(
                save_path, cv_mode=cv_mode, n_splits=n_splits, buffer_samples=buffer_samples
            )
        else:
            p = Path(save_path)
        if not p.exists():
            return None
        with p.open("rb") as f:
            obj = pickle.load(f)
        if verbose:
            print(f"[{label}] loaded: {p}")
        return obj

    @staticmethod
    def _save_result(
        save_path: Optional[str],
        result,
        label: str,
        verbose: bool,
        *,
        cv_mode: Optional[str] = None,
        n_splits: Optional[int] = None,
        buffer_samples: Optional[int] = None,
    ):
        if save_path is None:
            return
        if cv_mode is not None and n_splits is not None and buffer_samples is not None:
            p = StopDecodingRunner._path_with_cv_params(
                save_path, cv_mode=cv_mode, n_splits=n_splits, buffer_samples=buffer_samples
            )
        else:
            p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"[{label}] saved: {p}")

    def compute_canoncorr(
        self,
        *,
        varnames: Optional[Sequence[str]] = None,
        use_spike_history: Optional[bool] = None,
        filtwidth: int = 5,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        loaded = self._maybe_load_result(save_path, load_if_exists, "canoncorr", verbose)
        if loaded is not None:
            self.stats["canoncorr"] = loaded
            return loaded

        self._collect_data(exists_ok=True)
        y_df = self._get_numeric_target_df()
        if varnames is None:
            varnames = self._default_canoncorr_varnames()
        varnames = [v for v in varnames if v in y_df.columns]
        if len(varnames) == 0:
            raise ValueError("No valid canoncorr variables found in stop_feats_to_decode.")

        if verbose:
            print(f"[canoncorr] vars: {varnames}")
        x_task = y_df[varnames].to_numpy(dtype=float)
        y_neural = self._get_neural_matrix(use_spike_history=use_spike_history)

        out = decode_stops_utils.compute_canoncorr_block(
            x_task=x_task,
            y_neural=y_neural,
            dt=float(self.bin_width),
            filtwidth=int(filtwidth),
        )
        out["vars"] = list(varnames)
        self.stats["canoncorr"] = out
        self._save_result(save_path, out, "canoncorr", verbose)
        return out

    def regress_popreadout(
        self,
        *,
        varnames: Optional[Sequence[str]] = None,
        use_spike_history: Optional[bool] = None,
        fit_kernelwidth: bool = True,
        candidate_widths: Sequence[int] = tuple(range(1, 101, 5)),
        fixed_width: int = 25,
        n_splits: int = 5,
        cv_mode: str = "blocked_time_buffered",
        buffer_samples: int = 20,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        decodertype = "lineardecoder"
        loaded = self._maybe_load_result(
            save_path,
            load_if_exists,
            decodertype,
            verbose,
            cv_mode=cv_mode,
            n_splits=n_splits,
            buffer_samples=buffer_samples,
        )
        if loaded is not None:
            self.stats[decodertype] = loaded
            self.out = loaded
            print('Return loaded result')
            return
        else:
            print('Compute new result because no cached result found')

        self._collect_data(exists_ok=True)
        y_df = self._get_numeric_target_df()
        if varnames is None:
            varnames = self._default_readout_varnames()
        varnames = [v for v in varnames if v in y_df.columns]
        if len(varnames) == 0:
            raise ValueError("No valid readout variables found in stop_feats_to_decode.")

        neural = self._get_neural_matrix(use_spike_history=use_spike_history)
        groups = self.stop_meta_used["event_id"].values
        _, lengths = decode_stops_utils.build_group_lengths(groups)

        out: Dict = {}
        for v in varnames:
            if verbose:
                print(f"[{decodertype}] fitting {v} (CV n_splits={n_splits}, cv_mode={cv_mode})")
            x_true = y_df[v].to_numpy(dtype=float)
            x_true[np.isnan(x_true)] = 0.0

            if fit_kernelwidth:
                best = decode_stops_utils.tune_linear_decoder_cv(
                    y_neural=neural,
                    x_true=x_true,
                    lengths=lengths,
                    candidate_widths=candidate_widths,
                    n_splits=n_splits,
                    cv_mode=cv_mode,
                    buffer_samples=buffer_samples,
                )
            else:
                best = decode_stops_utils.fit_linear_decoder_cv(
                    y_neural=neural,
                    x_true=x_true,
                    lengths=lengths,
                    width=int(fixed_width),
                    n_splits=n_splits,
                    cv_mode=cv_mode,
                    buffer_samples=buffer_samples,
                )

            pred = best["pred"]
            out[v] = {
                "bestfiltwidth": int(best["width"]),
                "wts": best["wts"],
                "true": x_true,
                "pred": pred,
                "corr": decode_stops_utils.safe_corr(x_true, pred),
                "trials": {
                    "true": decode_stops_utils.split_by_lengths(x_true, lengths),
                    "pred": decode_stops_utils.split_by_lengths(pred, lengths),
                },
            }

        self.out["_cv_config"] = {
            "cv_mode": cv_mode,
            "n_splits": n_splits,
            "buffer_samples": buffer_samples,
        }
        self.stats[decodertype] = self.out
        self._save_result(
            save_path,
            self.out,
            decodertype,
            verbose,
            cv_mode=cv_mode,
            n_splits=n_splits,
            buffer_samples=buffer_samples,
        )
        return

    # ------------------------------------------------------------------
    # Plotting helpers (one-FF-style outputs)
    # ------------------------------------------------------------------
    def plot_canoncorr_coefficients(self, **plot_kwargs):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        block = self.stats.get("canoncorr")
        if block is None:
            raise ValueError("No canoncorr results found. Run compute_canoncorr() first.")
        plot_decode_stops.plot_canoncorr_coefficients(block, **plot_kwargs)

    def plot_decoder_parity(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_decode_stops.plot_decoder_parity(block, varnames=varnames, **plot_kwargs)

    def plot_decoder_correlation_bars(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_decode_stops.plot_decoder_correlation_bars(block, varnames=varnames, **plot_kwargs)

    def plot_single_trial_decoding_panel(
        self,
        *,
        trial_indices: Optional[Sequence[int]] = None,
        n_trials: int = 6,
        **plot_kwargs,
    ):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_decode_stops.plot_single_trial_decoding_panel(
            block,
            trial_indices=trial_indices,
            n_trials=n_trials,
            **plot_kwargs,
        )

    def plot_all_decoding_results(
        self,
        *,
        parity_varnames: Optional[Sequence[str]] = None,
        bar_varnames: Optional[Sequence[str]] = None,
        trial_indices: Optional[Sequence[int]] = None,
        n_trials: int = 6,
    ):
        from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import (
            plot_decode_stops,
        )

        plot_decode_stops.plot_all_decoding_results(
            canoncorr_block=self.stats.get("canoncorr"),
            readout_block=self.stats.get("lineardecoder"),
            parity_varnames=parity_varnames,
            bar_varnames=bar_varnames,
            trial_indices=trial_indices,
            n_trials=n_trials,
        )

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def _collect_data(self, exists_ok: bool = True):
        if exists_ok and self._load_design_matrices():
            print("[StopDecodingRunner] Using cached design matrices")
            return

        print("[StopDecodingRunner] Computing design matrices from scratch")
        (
            self.pn,
            self.stop_binned_spikes,
            self.stop_feats_to_decode,
            _offset_log,
            self.stop_meta_used,
            _stop_meta_groups,
        ) = decode_stops_design.assemble_stop_decoding_design(
            self.raw_data_folder_path,
            self.bin_width,
        )

        bin_df = spike_history.make_bin_df_from_stop_meta(self.stop_meta_used)
        (
            self.spike_data_w_history,
            _basis,
            _colnames,
            _meta_groups,
        ) = spike_history.build_design_with_spike_history_from_bins(
            spikes_df=self.pn.spikes_df,
            bin_df=bin_df,
            X_pruned=self.stop_binned_spikes,
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
            "decoding_outputs/stop_decoder_outputs",
        )

    def _get_design_matrix_paths(self):
        save_dir = Path(self._get_save_dir())
        return {
            "stop_binned_spikes": save_dir / "stop_binned_spikes.pkl",
            "spike_data_w_history": save_dir / "spike_data_w_history.pkl",
            "stop_feats_to_decode": save_dir / "stop_feats_to_decode.pkl",
            "stop_meta_used": save_dir / "stop_meta_used.pkl",
        }

    def _save_design_matrices(self):
        save_dir = Path(self._get_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = self._get_design_matrix_paths()
        data_to_save = {
            "stop_binned_spikes": self.stop_binned_spikes,
            "spike_data_w_history": self.spike_data_w_history,
            "stop_feats_to_decode": self.stop_feats_to_decode,
            "stop_meta_used": self.stop_meta_used,
        }
        for key, data in data_to_save.items():
            try:
                with open(paths[key], "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[StopDecodingRunner] Saved {key} -> {paths[key]}")
            except Exception as e:
                print(f"[StopDecodingRunner] WARNING save {key}: {type(e).__name__}: {e}")

    def _load_design_matrices(self):
        paths = self._get_design_matrix_paths()
        if not all(p.exists() for p in paths.values()):
            return False
        try:
            with open(paths["stop_binned_spikes"], "rb") as f:
                self.stop_binned_spikes = pickle.load(f)
            with open(paths["spike_data_w_history"], "rb") as f:
                self.spike_data_w_history = pickle.load(f)
            with open(paths["stop_feats_to_decode"], "rb") as f:
                self.stop_feats_to_decode = pickle.load(f)
            with open(paths["stop_meta_used"], "rb") as f:
                self.stop_meta_used = pickle.load(f)
            print("[StopDecodingRunner] Loaded cached design matrices")
            return True
        except Exception as e:
            print(f"[StopDecodingRunner] WARNING load matrices: {type(e).__name__}: {e}")
            return False
