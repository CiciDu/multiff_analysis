"""Base class for decoding runners with shared one-FF-style (CCA + linear readout) and CV decoding logic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    decode_stops_utils,
)
from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding import (
    cv_decoding,
)
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural.pn_decoding import (
    pn_decoding_model_specs,
)


class BaseOneFFStyleDecodingRunner:
    """
    Base class for decoding runners that support one-FF-style population decoding
    (CCA + linear readout). Subclasses must implement:
    - _get_target_df(): feature dataframe to decode
    - _get_groups(): group labels for CV (e.g. event_id)
    - _get_neural_matrix(): neural data array
    - _default_canoncorr_varnames(): default vars for CCA
    - _default_readout_varnames(): default vars for linear readout
    """

    def __init__(self, bin_width: float = 0.04):
        self.bin_width = bin_width
        self.stats: Dict = {}

    # ------------------------------------------------------------------
    # Abstract / override points (subclasses must implement)
    # ------------------------------------------------------------------
    def _get_target_df(self) -> pd.DataFrame:
        """Feature dataframe to decode (e.g. stop_feats_to_decode, behav_df, vis_feats_to_decode)."""
        raise NotImplementedError

    def _get_groups(self):
        """Group labels for CV (e.g. event_id or trial_ids)."""
        raise NotImplementedError

    def _get_neural_matrix(self, use_spike_history: Optional[bool] = None) -> np.ndarray:
        """Neural data matrix (samples x neurons)."""
        raise NotImplementedError

    def _default_canoncorr_varnames(self) -> List[str]:
        """Default variable names for canoncorr."""
        raise NotImplementedError

    def _default_readout_varnames(self) -> List[str]:
        """Default variable names for linear readout."""
        raise NotImplementedError

    def _get_save_dir(self) -> str:
        """Base save directory for outputs."""
        raise NotImplementedError

    def _get_cv_neural_X(self):
        """Neural data matrix for CV decoding (samples x features). Default: spike_data_w_history."""
        return self.spike_data_w_history

    def _runner_name(self) -> str:
        """Name for log messages. Override in subclass."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # CV decoding (run / run_cv_decoding)
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        n_splits: int = 5,
        save_dir=None,
        design_matrices_exists_ok: bool = True,
        model_specs=None,
        shuffle_mode: str = "none",
    ) -> pd.DataFrame:
        """Run CV decoding. Delegates to run_cv_decoding."""
        return self.run_cv_decoding(
            n_splits=n_splits,
            save_dir=save_dir,
            design_matrices_exists_ok=design_matrices_exists_ok,
            model_specs=model_specs,
            shuffle_mode=shuffle_mode,
        )

    def run_cv_decoding(
        self,
        *,
        n_splits: int = 5,
        save_dir=None,
        design_matrices_exists_ok: bool = True,
        model_specs=None,
        shuffle_mode: str = "none",
    ) -> pd.DataFrame:
        """Run cross-validated model-spec decoding."""
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
            print(f"[{self._runner_name()}] model_name: {model_name}")
            results_df = cv_decoding.run_cv_decoding(
                X=self._get_cv_neural_X(),
                y_df=self._get_target_df(),
                behav_features=None,
                groups=self._get_groups(),
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
    # Caching utilities (optional override)
    # ------------------------------------------------------------------
    def _get_design_matrix_paths(self) -> Mapping[str, Path]:
        """Paths for cached design matrices. Override in subclass."""
        raise NotImplementedError

    def _get_design_matrix_data(self) -> Mapping[str, Any]:
        """Data to save: key -> value. Override in subclass."""
        raise NotImplementedError

    def _get_design_matrix_key_to_attr(self) -> Mapping[str, str]:
        """Map path key -> attribute name for loading. Override in subclass."""
        raise NotImplementedError

    def _save_design_matrices(self) -> None:
        """Save design matrices to disk. Uses _get_design_matrix_paths and _get_design_matrix_data."""
        save_dir = Path(self._get_save_dir())
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = self._get_design_matrix_paths()
        data = self._get_design_matrix_data()
        name = self._runner_name()
        for key, path in paths.items():
            if key not in data:
                continue
            try:
                with open(path, "wb") as f:
                    pickle.dump(data[key], f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[{name}] Saved {key} -> {path}")
            except Exception as e:
                print(f"[{name}] WARNING save {key}: {type(e).__name__}: {e}")

    def _load_design_matrices(self) -> bool:
        """Load design matrices from disk. Returns True if successful."""
        paths = self._get_design_matrix_paths()
        key_to_attr = self._get_design_matrix_key_to_attr()
        if not all(paths[k].exists() for k in key_to_attr if k in paths):
            return False
        name = self._runner_name()
        try:
            for key, attr in key_to_attr.items():
                if key not in paths:
                    continue
                with open(paths[key], "rb") as f:
                    setattr(self, attr, pickle.load(f))
            print(f"[{name}] Loaded cached design matrices")
            return True
        except Exception as e:
            print(f"[{name}] WARNING load matrices: {type(e).__name__}: {e}")
            return False

    # ------------------------------------------------------------------
    # Shared one-FF-style logic
    # ------------------------------------------------------------------
    def _get_numeric_target_df(self) -> pd.DataFrame:
        """Filter target df to numeric columns (exclude const)."""
        y_df = self._get_target_df().copy()
        keep_cols = []
        for c in y_df.columns:
            if c == "const":
                continue
            if pd.api.types.is_numeric_dtype(y_df[c]) or pd.api.types.is_bool_dtype(y_df[c]):
                keep_cols.append(c)
        return y_df[keep_cols].astype(float)

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
        fit_kernelwidth: bool = True,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """Run one-FF-style decoding: CCA + linear readout. Requires _collect_data implemented."""
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
            fit_kernelwidth=fit_kernelwidth,
            save_path=str(save_dir / "lineardecoder.pkl"),
            load_if_exists=load_if_exists,
            verbose=verbose,
        )
        return {"canoncorr": canoncorr, "readout": readout, "stats": self.stats}

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
            p = BaseOneFFStyleDecodingRunner._path_with_cv_params(
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
            p = BaseOneFFStyleDecodingRunner._path_with_cv_params(
                save_path, cv_mode=cv_mode, n_splits=n_splits, buffer_samples=buffer_samples
            )
        else:
            p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"[{label}] saved: {p}")

    def _target_df_error_msg(self) -> str:
        """Override in subclass for clearer error messages."""
        return "target features dataframe"

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
            raise ValueError(
                f"No valid canoncorr variables found in {self._target_df_error_msg()}."
            )

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
        save_predictions: bool = False,
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
            if verbose:
                print(f"[{decodertype}] loaded from cache")
            return loaded

        if verbose:
            print(f"[{decodertype}] computing (no cached result found)")
        self._collect_data(exists_ok=True)
        y_df = self._get_numeric_target_df()
        if varnames is None:
            varnames = self._default_readout_varnames()
        varnames = [v for v in varnames if v in y_df.columns]
        if len(varnames) == 0:
            raise ValueError(
                f"No valid readout variables found in {self._target_df_error_msg()}."
            )

        neural = self._get_neural_matrix(use_spike_history=use_spike_history)
        groups = self._get_groups()
        groups = np.asarray(groups)
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
            widths_used = (
                list(candidate_widths) if fit_kernelwidth else [int(fixed_width)]
            )
            entry = {
                "bestfiltwidth": int(best["width"]),
                "candidate_widths": widths_used,
                "wts": best["wts"],
                "corr": decode_stops_utils.safe_corr(x_true, pred),
            }
            if save_predictions:
                entry["true"] = x_true
                entry["pred"] = pred
                entry["trials"] = {
                    "true": decode_stops_utils.split_by_lengths(x_true, lengths),
                    "pred": decode_stops_utils.split_by_lengths(pred, lengths),
                }
            out[v] = entry

        out["_cv_config"] = {
            "cv_mode": cv_mode,
            "n_splits": n_splits,
            "buffer_samples": buffer_samples,
        }
        self.stats[decodertype] = out
        self._save_result(
            save_path,
            out,
            decodertype,
            verbose,
            cv_mode=cv_mode,
            n_splits=n_splits,
            buffer_samples=buffer_samples,
        )
        return out
