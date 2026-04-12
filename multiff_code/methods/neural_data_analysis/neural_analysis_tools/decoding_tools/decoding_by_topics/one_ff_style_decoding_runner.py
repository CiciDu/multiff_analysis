"""Base class for decoding runners with shared one-FF-style (CCA + linear readout) and CV decoding logic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    decode_stops_utils, plot_decoding_utils
)

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import plot_one_ff_decoding

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_by_topics import one_ff_style_utils

class OneFFStyleDecodingRunner:


    # ------------------------------------------------------------------
    # Shared one-FF-style logic
    # ------------------------------------------------------------------
    def _get_numeric_target_df(self) -> pd.DataFrame:
        """Filter target df to numeric columns (exclude const)."""
        y_df = self.get_target_df().copy()
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
        """Run one-FF-style decoding: CCA + linear readout. Requires collect_data implemented."""
        self.collect_data(exists_ok=design_matrices_exists_ok)
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
        print('Finished running one-FF-style decoding')
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
            p = OneFFStyleDecodingRunner._path_with_cv_params(
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
            p = OneFFStyleDecodingRunner._path_with_cv_params(
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

        self.collect_data(exists_ok=True)
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
        groups = self._get_groups()
        trial_full = np.asarray(groups) if groups is not None else None
        y_neural = self._get_neural_matrix_for_decoding(
            neural_smooth_width=int(filtwidth),
            trial_idx=trial_full,
        )
        n_smooth = (
            self._neural_ncols_after_pca()
            if getattr(self, "use_spike_history", False)
            else None
        )

        out = one_ff_style_utils.compute_canoncorr_block(
            x_task=x_task,
            y_neural=y_neural,
            dt=float(self.bin_width),
            filtwidth=0,
            neural_cols_to_smooth=n_smooth,
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
        candidate_widths: Sequence[int] = tuple(range(1, 21, 1)),
        fixed_width: int = 25,
        n_splits: int = 5,
        inner_cv_splits: int = 5,
        cv_mode: str = "blocked_time_buffered",  # 'blocked_time_buffered', 'blocked_time', 'group_kfold', 'kfold'
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
        self.collect_data(exists_ok=True)
        y_df = self._get_numeric_target_df()
        if varnames is None:
            varnames = self._default_readout_varnames()
        varnames = [v for v in varnames if v in y_df.columns]
        if len(varnames) == 0:
            raise ValueError(
                f"No valid readout variables found in {self._target_df_error_msg()}."
            )

        groups = self._get_groups()
        groups = np.asarray(groups)
        trial_full = groups
        if fit_kernelwidth:
            neural = self._get_neural_matrix_for_decoding(
                neural_smooth_width=0,
                trial_idx=trial_full,
            )
        else:
            neural = self._get_neural_matrix_for_decoding(
                neural_smooth_width=int(fixed_width),
                trial_idx=trial_full,
            )
        _, lengths = one_ff_style_utils.build_group_lengths(groups)
        n_smooth = (
            self._neural_ncols_after_pca()
            if getattr(self, "use_spike_history", False)
            else None
        )

        out: Dict = {}
        for v in varnames:
            if verbose:
                print(f"[{decodertype}] fitting {v} (CV n_splits={n_splits}, cv_mode={cv_mode})")
            x_true = y_df[v].to_numpy(dtype=float)
            x_true[np.isnan(x_true)] = 0.0
            if fit_kernelwidth:
                from neural_data_analysis.neural_analysis_tools.decoding_tools.general_decoding.cv_decoding import (
                    _build_folds,
                )

                N = len(neural)
                outer_splits = _build_folds(
                    N,
                    n_splits=n_splits,
                    groups=groups,
                    cv_splitter=cv_mode,
                    random_state=0,
                )

                pred = np.full(N, np.nan)
                widths_per_fold = []
                wts_per_fold = []
                fold_tuning_info = []

                for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
                    train_idx = np.asarray(train_idx, dtype=int)
                    test_idx = np.asarray(test_idx, dtype=int)

                    _, lengths_train = one_ff_style_utils.build_group_lengths(groups[train_idx])

                    if getattr(self, "pca_n_components", None) is not None:
                        # Width must vary presmooth(raw) → PCA, not smooth(PCA components).
                        if not len(candidate_widths):
                            raise ValueError(
                                "candidate_widths must be non-empty when "
                                "fit_kernelwidth=True and PCA is enabled."
                            )
                        best_err = float("inf")
                        best_w = int(candidate_widths[0])
                        best_inner = None
                        for w in candidate_widths:
                            neu_w = self._get_neural_matrix_for_decoding(
                                neural_smooth_width=int(w),
                                trial_idx=trial_full,
                            )[train_idx]
                            cur = one_ff_style_utils.fit_linear_decoder_cv(
                                y_neural=neu_w,
                                x_true=x_true[train_idx],
                                lengths=lengths_train,
                                width=0,
                                n_splits=inner_cv_splits,
                                cv_mode=cv_mode,
                                buffer_samples=buffer_samples,
                                neural_cols_to_smooth=None,
                            )
                            if cur["error"] < best_err:
                                best_err = float(cur["error"])
                                best_w = int(w)
                                best_inner = cur
                        best = dict(best_inner) if best_inner is not None else {}
                        if best_inner is not None:
                            best["width"] = best_w
                    else:
                        best = one_ff_style_utils.tune_linear_decoder_cv(
                            y_neural=neural[train_idx],
                            x_true=x_true[train_idx],
                            lengths=lengths_train,
                            candidate_widths=candidate_widths,
                            n_splits=inner_cv_splits,
                            cv_mode=cv_mode,
                            buffer_samples=buffer_samples,
                            neural_cols_to_smooth=n_smooth,
                        )

                    best_width = int(best["width"])
                    widths_per_fold.append(best_width)
                    fold_tuning_info.append(best.get("width_scores", {}))

                    wts = np.asarray(best.get("wts"))
                    wts_per_fold.append(wts)

                    # Smooth raw (then PCA / hist) at chosen width; align rows to this fold
                    X_tr = self._get_neural_matrix_for_decoding(
                        neural_smooth_width=int(best_width),
                        trial_idx=trial_full,
                    )[train_idx]
                    X_te = self._get_neural_matrix_for_decoding(
                        neural_smooth_width=int(best_width),
                        trial_idx=trial_full,
                    )[test_idx]

                    # Fit linear weights on training set (fallback to best['wts'] if numerical issues)
                    try:
                        coef, *_ = np.linalg.lstsq(X_tr, x_true[train_idx], rcond=None)
                    except Exception:
                        coef = wts

                    coef = np.asarray(coef).reshape(-1)
                    pred_test = X_te.dot(coef)
                    pred[test_idx] = pred_test

                widths_used = list(candidate_widths)
                # Representative width for reporting
                rep_width = int(np.round(float(np.nanmedian(widths_per_fold)))) if widths_per_fold else int(fixed_width)
                entry = {
                    "bestfiltwidth": rep_width,
                    "candidate_widths": widths_used,
                    "wts": wts_per_fold,
                    "corr": one_ff_style_utils.safe_corr(x_true, pred),
                }
                if save_predictions:
                    entry["true"] = x_true
                    entry["pred"] = pred
                    entry["trials"] = {
                        "true": one_ff_style_utils.split_by_lengths(x_true, lengths),
                        "pred": one_ff_style_utils.split_by_lengths(pred, lengths),
                    }
                entry["fold_tuning_info"] = fold_tuning_info

                out[v] = entry
                continue
            else:
                best = one_ff_style_utils.fit_linear_decoder_cv(
                    y_neural=neural,
                    x_true=x_true,
                    lengths=lengths,
                    width=int(fixed_width),
                    n_splits=n_splits,
                    cv_mode=cv_mode,
                    buffer_samples=buffer_samples,
                    neural_cols_to_smooth=n_smooth,
                )

            pred = best["pred"]
            widths_used = (
                list(candidate_widths) if fit_kernelwidth else [int(fixed_width)]
            )
            entry = {
                "bestfiltwidth": int(best["width"]),
                "candidate_widths": widths_used,
                "wts": best["wts"],
                "corr": one_ff_style_utils.safe_corr(x_true, pred),
            }
            if save_predictions:
                entry["true"] = x_true
                entry["pred"] = pred
                entry["trials"] = {
                    "true": one_ff_style_utils.split_by_lengths(x_true, lengths),
                    "pred": one_ff_style_utils.split_by_lengths(pred, lengths),
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


    # ------------------------------------------------------------------
    # Plotting helpers (one-FF-style outputs)
    # ------------------------------------------------------------------
    def plot_canoncorr_coefficients(self, **plot_kwargs):
        block = self.stats.get("canoncorr")
        if block is None:
            raise ValueError("No canoncorr results found. Run compute_canoncorr() first.")
        plot_one_ff_decoding.plot_canoncorr_coefficients(block, **plot_kwargs)

    def plot_decoder_parity(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):
        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_one_ff_decoding.plot_decoder_parity(block, varnames=varnames, **plot_kwargs)

    def plot_decoder_correlation_bars(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):
        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_one_ff_decoding.plot_decoder_correlation_bars(block, varnames=varnames, **plot_kwargs)


    def plot_single_trial_decoding_panel(
        self,
        *,
        trial_indices: Optional[Sequence[int]] = None,
        n_trials: int = 6,
        **plot_kwargs,
    ):

        block = self.stats.get("lineardecoder")
        if block is None:
            raise ValueError("No lineardecoder results found. Run regress_popreadout() first.")
        plot_decoding_utils.plot_single_trial_decoding_panel(
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

        plot_decoding_utils.plot_all_decoding_results(
            canoncorr_block=self.stats.get("canoncorr"),
            readout_block=self.stats.get("lineardecoder"),
            parity_varnames=parity_varnames,
            bar_varnames=bar_varnames,
            trial_indices=trial_indices,
            n_trials=n_trials,
        )


    def extract_one_ff_corr_df(self, block_key='lineardecoder'):
        '''
        Extract decoder correlation results from self.stats[block_key]
        and return as a sorted dataframe.

        Returns
        -------
        df : pandas.DataFrame
            Columns:
            - variable
            - corr
        '''

        readout_block = self.stats.get(block_key)

        if readout_block is None:
            raise ValueError(f'No block "{block_key}" found in stats.')

        rows = []

        for k, v in readout_block.items():
            if isinstance(v, dict) and ('corr' in v):
                rows.append({
                    'variable': k,
                    'corr': float(v.get('corr', np.nan))
                })

        if len(rows) == 0:
            raise ValueError('No decoder correlations found in readout block.')

        df = pd.DataFrame(rows)

        df = df.sort_values(
            'corr',
            key=lambda x: np.nan_to_num(x, nan=-np.inf),
            ascending=False
        ).reset_index(drop=True)

        self.one_ff_corr_df = df.copy()
