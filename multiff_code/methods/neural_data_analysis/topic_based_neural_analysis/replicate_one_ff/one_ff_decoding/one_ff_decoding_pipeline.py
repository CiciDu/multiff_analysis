from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA

from neural_data_analysis.neural_analysis_tools.decoding_tools.decoding_helpers import (
    decode_stops_utils,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import (
    one_ff_parameters,
    one_ff_pipeline,
    population_analysis_utils,
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_decoding import plot_one_ff_decoding


DEFAULT_CANONCORR_VARS = ["v", "w", "d", "phi", "r_targ", "theta_targ"]
DEFAULT_READOUT_VARS = ["v", "w", "d", "phi", "r_targ", "theta_targ"]


@dataclass
class DecodingDataBundle:
    x_by_var: Dict[str, np.ndarray]
    y_raw: np.ndarray
    trial_lengths: List[int]
    selected_trials: np.ndarray
    trial_ts: List[np.ndarray]
    trial_events: List[object]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _get_attr(obj, name: str, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _gen_traj_from_w_v(w_deg_per_s: np.ndarray, v: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    phi_deg = np.cumsum(np.asarray(w_deg_per_s, dtype=float)) * dt
    phi_rad = np.deg2rad(phi_deg)
    dx = np.asarray(v, dtype=float) * np.cos(phi_rad) * dt
    dy = np.asarray(v, dtype=float) * np.sin(phi_rad) * dt
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    return x, y


class OneFFDecodingRunner:
    """
    Python pipeline for one-FF population decoding, adapted from AnalysePopulation.m.

    This runner provides:
    - canonical correlation between selected behavioral variables and population response
    - linear population readout (OLS with optional smoothing-width tuning)
    - trial-wise decoded trajectories and error summaries
    """

    def __init__(
        self,
        *,
        session_num: int = 0,
        prs=None,
        mat_path: str = "all_monkey_data/one_ff_data/sessions_python.mat",
        output_root: str = "all_monkey_data/one_ff_data/decoding",
    ):
        self.session_num = session_num
        self.prs = prs if prs is not None else one_ff_parameters.default_prs()
        self.mat_path = mat_path
        self.output_root = Path(output_root)

        self.data_obj: Optional[one_ff_pipeline.OneFFSessionData] = None
        self.stats: Dict = {}
        # Exposed intermediate state for debugging/inspection
        self.last_bundle: Optional[DecodingDataBundle] = None
        self.current_varnames: List[str] = []
        self.x_matrix: Optional[np.ndarray] = None
        self.y_raw: Optional[np.ndarray] = None
        self.y_smooth: Optional[np.ndarray] = None
        self.y_rate: Optional[np.ndarray] = None
        self.filtwidth: Optional[int] = None
        self.cca_model: Optional[CCA] = None

        self.readout_bundle: Optional[DecodingDataBundle] = None
        self.readout_varnames: List[str] = []
        self.readout_y_raw: Optional[np.ndarray] = None
        self.readout_candidate_widths: Optional[np.ndarray] = None
        self.readout_y_smooth_by_width: Dict[int, np.ndarray] = {}
        self.readout_x_true_by_var: Dict[str, np.ndarray] = {}
        self.readout_y_fit_by_var: Dict[str, np.ndarray] = {}
        self.readout_pred_by_var: Dict[str, np.ndarray] = {}

        # Cache bundles by variable selection + timing context.
        self._bundle_cache: Dict[Tuple, DecodingDataBundle] = {}
        self._bundle_cache_last_key: Optional[Tuple] = None

    def _ensure_data(self) -> None:
        if self.data_obj is not None:
            return
        self.data_obj = one_ff_pipeline.OneFFSessionData(
            mat_path=self.mat_path,
            prs=self.prs,
            session_num=self.session_num,
        )

    def _trial_bin_indices(self, tr, pretrial: float, posttrial: float) -> np.ndarray:
        ts = tr.continuous.ts
        ts_trim = ts[1:-1]
        t_start, t_stop = population_analysis_utils.full_time_window(
            tr, pretrial=pretrial, posttrial=posttrial
        )
        mask = (ts_trim > t_start) & (ts_trim < t_stop)
        return np.where(mask)[0] + 1

    def _extract_var_from_trial(self, tr, covs: Dict[str, np.ndarray], varname: str, dt: float) -> np.ndarray:
        if varname in covs:
            return np.asarray(covs[varname], dtype=float)
        if varname == "dv":
            v = np.asarray(covs["v"], dtype=float)
            return np.concatenate([[0.0], np.diff(v) / dt])
        if varname == "dw":
            w = np.asarray(covs["w"], dtype=float)
            return np.concatenate([[0.0], np.diff(w) / dt])
        raise ValueError(f"Unsupported decode/canoncorr variable: {varname}")

    def _bundle_cache_key(self, varnames: Sequence[str]) -> Tuple:
        pretrial = float(_get_attr(self.prs, "pretrial", 0.5))
        posttrial = float(_get_attr(self.prs, "posttrial", 0.5))
        return (
            tuple(varnames),
            float(self.prs.dt),
            pretrial,
            posttrial,
            int(self.session_num),
            str(self.mat_path),
        )

    def _build_population_bundle(self, *, varnames: Sequence[str]) -> DecodingDataBundle:
        varnames = list(varnames)
        cache_key = self._bundle_cache_key(varnames)
        cached = self._bundle_cache.get(cache_key)
        if cached is not None:
            self.last_bundle = cached
            self._bundle_cache_last_key = cache_key
            return cached

        self._ensure_data()
        pretrial = float(_get_attr(self.prs, "pretrial", 0.5))
        posttrial = float(_get_attr(self.prs, "posttrial", 0.5))
        sel_trials = np.asarray(self.data_obj.sel_trials, dtype=int)

        # Use shared OneFFSessionData concatenation path for aligned covariates/spikes.
        required_covs = sorted({("v" if v == "dv" else "w" if v == "dw" else v) for v in varnames})
        self.data_obj.compute_covariates(required_covs)
        self.data_obj.compute_spike_counts()

        trial_id_vec = np.asarray(self.data_obj.covariate_trial_ids, dtype=int)
        trial_lengths = [int(np.sum(trial_id_vec == tid)) for tid in sel_trials]

        cov_concat = {k: np.asarray(v, dtype=float) for k, v in self.data_obj.covariates.items()}
        x_by_var: Dict[str, np.ndarray] = {}
        for v in varnames:
            if v in cov_concat:
                x_by_var[v] = cov_concat[v]
            elif v == "dv":
                v_trials = population_analysis_utils.deconcatenate_trials(cov_concat["v"], trial_lengths)
                x_by_var[v] = np.concatenate(
                    [np.concatenate([[0.0], np.diff(vt) / float(self.prs.dt)]) for vt in v_trials]
                )
            elif v == "dw":
                w_trials = population_analysis_utils.deconcatenate_trials(cov_concat["w"], trial_lengths)
                x_by_var[v] = np.concatenate(
                    [np.concatenate([[0.0], np.diff(wt) / float(self.prs.dt)]) for wt in w_trials]
                )
            else:
                raise ValueError(f"Unsupported decode/canoncorr variable: {v}")

        trial_ts: List[np.ndarray] = []
        trial_events: List[object] = []
        for tid in sel_trials:
            tr = self.data_obj.all_trials[tid]
            idx = self._trial_bin_indices(tr, pretrial=pretrial, posttrial=posttrial)
            trial_ts.append(np.asarray(tr.continuous.ts[idx], dtype=float))
            trial_events.append(tr.events)

        y_raw = np.asarray(self.data_obj.Y, dtype=float)

        print('Made new DecodingDataBundle instance.')
        bundle = DecodingDataBundle(
            x_by_var=x_by_var,
            y_raw=y_raw,
            trial_lengths=trial_lengths,
            selected_trials=sel_trials,
            trial_ts=trial_ts,
            trial_events=trial_events,
        )
        self._bundle_cache[cache_key] = bundle
        self.last_bundle = bundle
        self._bundle_cache_last_key = cache_key
        return bundle

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

    def _maybe_load_result(
        self,
        *,
        save_path: Optional[str],
        load_if_exists: bool,
        label: str,
        verbose: bool,
        cv_mode: Optional[str] = None,
        n_splits: Optional[int] = None,
        buffer_samples: Optional[int] = None,
    ) -> Optional[Dict]:
        if (not load_if_exists) or save_path is None:
            return None
        if cv_mode is not None and n_splits is not None and buffer_samples is not None:
            path = self._path_with_cv_params(
                save_path, cv_mode=cv_mode, n_splits=n_splits, buffer_samples=buffer_samples
            )
        else:
            path = Path(save_path)
        if not path.exists():
            return None
        with path.open("rb") as f:
            loaded = pickle.load(f)
        if verbose:
            print(f"[{label}] Loaded saved result: {path}")
        return loaded

    def _save_result(
        self,
        *,
        save_path: Optional[str],
        result: Dict,
        label: str,
        verbose: bool,
        cv_mode: Optional[str] = None,
        n_splits: Optional[int] = None,
        buffer_samples: Optional[int] = None,
    ) -> None:
        if save_path is None:
            return
        if cv_mode is not None and n_splits is not None and buffer_samples is not None:
            path = self._path_with_cv_params(
                save_path, cv_mode=cv_mode, n_splits=n_splits, buffer_samples=buffer_samples
            )
        else:
            path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(result, f)
        if verbose:
            print(f"[{label}] Saved result: {path}")

    def _default_save_dir(self) -> Path:
        """Default save directory: output_root/session_N."""
        return self.output_root / f"session_{self.session_num}"

    def compute_canoncorr(
        self,
        *,
        varnames: Optional[Sequence[str]] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        if varnames is None:
            varnames = _get_attr(self.prs, "canoncorr_varname", DEFAULT_CANONCORR_VARS)
        varnames = list(varnames)

        if save_path is None:
            save_dir = self._default_save_dir()
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / "canoncorr.pkl")

        loaded = self._maybe_load_result(
            save_path=save_path,
            load_if_exists=load_if_exists,
            label="canoncorr",
            verbose=verbose,
        )
        if loaded is not None:
            self.stats.setdefault("trialtype", {}).setdefault("all", {})["canoncorr"] = loaded
            return loaded

        if verbose:
            print(f"[canoncorr] Building data for vars: {varnames}")
        bundle = self._build_population_bundle(varnames=varnames)
        x = np.column_stack([bundle.x_by_var[v] for v in varnames]).astype(float)
        x[np.isnan(x)] = 0.0

        filtwidth = int(_get_attr(self.prs, "neuralfiltwidth", _get_attr(self.prs, "neural_filtwidth", 10)))
        y_smooth = population_analysis_utils.smooth_signal(bundle.y_raw, max(filtwidth, 1))
        y_rate = y_smooth / float(self.prs.dt)

        n_comp = int(min(x.shape[1], y_rate.shape[1]))
        cca = CCA(n_components=max(1, n_comp), max_iter=2000)
        x_c, y_c = cca.fit_transform(x, y_rate)

        # Save intermediate state
        self.last_bundle = bundle
        self.current_varnames = varnames
        self.x_matrix = x
        self.y_raw = bundle.y_raw
        self.y_smooth = y_smooth
        self.y_rate = y_rate
        self.filtwidth = filtwidth
        self.cca_model = cca

        coeff = np.array([_safe_corr(x_c[:, i], y_c[:, i]) for i in range(x_c.shape[1])], dtype=float)
        coeff_sq = coeff ** 2
        dimensionality = (
            float((coeff_sq.sum() ** 2) / np.maximum((coeff_sq ** 2).sum(), 1e-12))
            if coeff_sq.size > 0
            else np.nan
        )

        out = {
            "vars": list(varnames),
            "stim": x_c,
            "resp": y_c,
            "coeff": coeff,
            "dimensionality": dimensionality,
            "responsecorr_raw": np.corrcoef(bundle.y_raw, rowvar=False),
            "responsecorr_smooth": np.corrcoef(y_smooth, rowvar=False),
        }
        self.stats.setdefault("trialtype", {}).setdefault("all", {})["canoncorr"] = out
        self._save_result(
            save_path=save_path,
            result=out,
            label="canoncorr",
            verbose=verbose,
        )
        if verbose:
            print("[canoncorr] Completed.")
        return out

    def regress_popreadout(
        self,
        *,
        varnames: Optional[Sequence[str]] = None,
        n_splits: Optional[int] = None,
        cv_mode: Optional[str] = None,
        buffer_samples: Optional[int] = None,
        save_path: Optional[str] = None,
        load_if_exists: bool = True,
        verbose: bool = True,
    ) -> Dict:
        if varnames is None:
            varnames = _get_attr(self.prs, "readout_varname", DEFAULT_READOUT_VARS)
        varnames = list(varnames)
        decodertype = str(_get_attr(self.prs, "decodertype", "lineardecoder"))
        if n_splits is None:
            n_splits = int(_get_attr(self.prs, "lineardecoder_n_splits", 5))
        if cv_mode is None:
            cv_mode = str(_get_attr(self.prs, "lineardecoder_cv_mode", "group_kfold"))
        if buffer_samples is None:
            buffer_samples = int(_get_attr(self.prs, "lineardecoder_buffer_samples", 20))

        if save_path is None:
            save_dir = self._default_save_dir()
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / "lineardecoder.pkl")

        loaded = self._maybe_load_result(
            save_path=save_path,
            load_if_exists=load_if_exists,
            label=decodertype,
            verbose=verbose,
            cv_mode=cv_mode,
            n_splits=n_splits,
            buffer_samples=buffer_samples,
        )
        if loaded is not None:
            self.stats.setdefault(decodertype, {}).update(loaded)
            self.out = loaded
            self._ensure_trajectories_and_flat_truth(decodertype)
            if verbose:
                cfg = loaded.get("_cv_config", {})
                print(f"[{decodertype}] Loaded (cv_mode={cfg.get('cv_mode', '?')}, n_splits={cfg.get('n_splits', '?')})")
            return
        else:
            pass  # Compute below

        if verbose:
            print(f"[{decodertype}] Building data for vars: {varnames} (CV n_splits={n_splits}, cv_mode={cv_mode})")
        bundle = self._build_population_bundle(varnames=varnames)
        y_raw = bundle.y_raw
        filtwidth_default = int(
            _get_attr(self.prs, "neuralfiltwidth", _get_attr(self.prs, "neural_filtwidth", 10))
        )
        fit_kernel = bool(_get_attr(self.prs, "lineardecoder_fitkernelwidth", True))
        candidate_widths = np.arange(1, 21, 1) if fit_kernel else None
        fixed_width = int(max(1, 5 * filtwidth_default))

        # Save intermediate state
        self.readout_bundle = bundle
        self.readout_varnames = varnames
        self.readout_y_raw = y_raw
        self.readout_candidate_widths = candidate_widths
        self.readout_y_smooth_by_width = {}
        self.readout_x_true_by_var = {}
        self.readout_y_fit_by_var = {}
        self.readout_pred_by_var = {}

        results: Dict[str, Dict] = {}
        for var in varnames:
            if verbose:
                print(f"[{decodertype}] Fitting decoder for: {var}")
            x_true = np.asarray(bundle.x_by_var[var], dtype=float).copy()
            x_true[np.isnan(x_true)] = 0.0

            if var in {"dv", "dw"}:
                x_true = population_analysis_utils.smooth_signal(x_true, max(filtwidth_default, 1))

            if fit_kernel and candidate_widths is not None:
                best = decode_stops_utils.tune_linear_decoder_cv(
                    y_neural=y_raw,
                    x_true=x_true,
                    lengths=bundle.trial_lengths,
                    candidate_widths=candidate_widths,
                    n_splits=n_splits,
                    cv_mode=cv_mode,
                    buffer_samples=buffer_samples,
                )
            else:
                best = decode_stops_utils.fit_linear_decoder_cv(
                    y_neural=y_raw,
                    x_true=x_true,
                    lengths=bundle.trial_lengths,
                    width=fixed_width,
                    n_splits=n_splits,
                    cv_mode=cv_mode,
                    buffer_samples=buffer_samples,
                )

            best_width = int(best["width"])
            best_wts = best["wts"]
            best_pred = best["pred"]
            y_fit = population_analysis_utils.smooth_signal(y_raw, best_width)

            self.readout_y_smooth_by_width[int(best_width)] = y_fit
            self.readout_x_true_by_var[var] = x_true
            self.readout_y_fit_by_var[var] = y_fit
            self.readout_pred_by_var[var] = best_pred

            true_trials = population_analysis_utils.deconcatenate_trials(
                x_true, bundle.trial_lengths
            )
            pred_trials = population_analysis_utils.deconcatenate_trials(
                best_pred, bundle.trial_lengths
            )

            results[var] = {
                "bestfiltwidth": int(best_width),
                "wts": best_wts,
                "corr": decode_stops_utils.safe_corr(x_true, best_pred),
                "trials": {
                    "true": true_trials,
                    "pred": pred_trials,
                },
            }

        results["_cv_config"] = {
            "cv_mode": cv_mode,
            "n_splits": n_splits,
            "buffer_samples": buffer_samples,
        }
        self.stats.setdefault(decodertype, {}).update(results)
        self._populate_decoded_trajectory_and_errors(
            decodertype=decodertype,
            trial_ts=bundle.trial_ts,
            trial_events=bundle.trial_events,
        )
        # Save full readout block (including trajectories) so loading has everything
        to_save = dict(self.stats[decodertype])
        self._save_result(
            save_path=save_path,
            result=to_save,
            label=decodertype,
            verbose=verbose,
            cv_mode=cv_mode,
            n_splits=n_splits,
            buffer_samples=buffer_samples,
        )
        if verbose:
            print(f"[{decodertype}] Completed.")
        return results

    def _ensure_trajectories_and_flat_truth(self, decodertype: str) -> None:
        """Ensure trajectories exist when missing (e.g. from old cache)."""
        dec = self.stats.get(decodertype, {})
        # Populate trajectory from v,w when missing (e.g. from old cache)
        if "xt_from_vw" not in dec or not dec.get("xt_from_vw", {}).get("trials", {}).get("pred"):
            if {"v", "w"}.issubset(dec.keys()):
                xt_trials = []
                yt_trials = []
                for vp, wp in zip(dec["v"]["trials"]["pred"], dec["w"]["trials"]["pred"]):
                    x, y = _gen_traj_from_w_v(
                        np.asarray(wp), np.asarray(vp), dt=float(self.prs.dt)
                    )
                    xt_trials.append(x)
                    yt_trials.append(y)
                dec.setdefault("xt_from_vw", {}).setdefault("trials", {})["pred"] = xt_trials
                dec.setdefault("yt_from_vw", {}).setdefault("trials", {})["pred"] = yt_trials

    def _populate_decoded_trajectory_and_errors(
        self,
        *,
        decodertype: str,
        trial_ts: Sequence[np.ndarray],
        trial_events: Sequence[object],
    ) -> None:
        dec = self.stats.get(decodertype, {})
        required_for_vw = {"v", "w"}
        required_for_dp = {"d", "phi"}

        if required_for_vw.issubset(dec.keys()):
            xt_trials = []
            yt_trials = []
            for v_pred, w_pred in zip(dec["v"]["trials"]["pred"], dec["w"]["trials"]["pred"]):
                x, y = _gen_traj_from_w_v(w_pred, v_pred, dt=float(self.prs.dt))
                xt_trials.append(x)
                yt_trials.append(y)
            dec.setdefault("xt_from_vw", {}).setdefault("trials", {})["pred"] = xt_trials
            dec.setdefault("yt_from_vw", {}).setdefault("trials", {})["pred"] = yt_trials

        if required_for_dp.issubset(dec.keys()):
            xt_trials = []
            yt_trials = []
            for d_pred, phi_pred in zip(dec["d"]["trials"]["pred"], dec["phi"]["trials"]["pred"]):
                if d_pred.size <= 1 or phi_pred.size <= 1:
                    xt_trials.append(np.array([]))
                    yt_trials.append(np.array([]))
                    continue
                v_from_d = np.diff(d_pred) / float(self.prs.dt)
                w_from_phi = np.diff(phi_pred) / float(self.prs.dt)
                x, y = _gen_traj_from_w_v(w_from_phi, v_from_d, dt=float(self.prs.dt))
                xt_trials.append(x)
                yt_trials.append(y)
            dec.setdefault("xt", {}).setdefault("trials", {})["pred"] = xt_trials
            dec.setdefault("yt", {}).setdefault("trials", {})["pred"] = yt_trials

        needed_err = {"r_targ", "theta_targ", "v", "w"}
        if needed_err.issubset(dec.keys()):
            err = {"r_targ": [], "theta_targ": [], "v": [], "w": []}
            for i, (ts, ev) in enumerate(zip(trial_ts, trial_events)):
                if ts.size == 0:
                    continue
                start_idx = int(np.searchsorted(ts, 0.0, side="left"))
                stop_idx = int(np.searchsorted(ts, float(ev.t_stop), side="left"))
                stop_idx = min(max(stop_idx, 0), ts.size - 1)
                if stop_idx < start_idx:
                    continue

                err["r_targ"].append(
                    dec["r_targ"]["trials"]["true"][i][stop_idx]
                    - dec["r_targ"]["trials"]["pred"][i][stop_idx]
                )
                err["theta_targ"].append(
                    dec["theta_targ"]["trials"]["true"][i][stop_idx]
                    - dec["theta_targ"]["trials"]["pred"][i][stop_idx]
                )
                err["v"].append(
                    float(
                        np.nanmean(
                            dec["v"]["trials"]["true"][i][start_idx : stop_idx + 1]
                            - dec["v"]["trials"]["pred"][i][start_idx : stop_idx + 1]
                        )
                    )
                )
                err["w"].append(
                    float(
                        np.nanmean(
                            dec["w"]["trials"]["true"][i][start_idx : stop_idx + 1]
                            - dec["w"]["trials"]["pred"][i][start_idx : stop_idx + 1]
                        )
                    )
                )
            self.stats["error_lineardecoder"] = {k: np.asarray(v) for k, v in err.items()}

    def plot_canoncorr_coefficients(self, **plot_kwargs):
        canoncorr = self.stats.get("trialtype", {}).get("all", {}).get("canoncorr")
        if canoncorr is None:
            raise ValueError("No canoncorr results found. Run compute_canoncorr() first.")
        return plot_one_ff_decoding.plot_canoncorr_coefficients(canoncorr, **plot_kwargs)

    def plot_decoder_parity(self, *, varnames: Optional[Sequence[str]] = None, **plot_kwargs):
        decodertype = str(_get_attr(self.prs, "decodertype", "lineardecoder"))
        readout = self.stats.get(decodertype)
        if readout is None:
            raise ValueError("No decoder results found. Run regress_popreadout() first.")
        return plot_one_ff_decoding.plot_decoder_parity(
            readout,
            varnames=varnames,
            **plot_kwargs,
        )

    def plot_decoded_trajectory(
        self,
        *,
        trial_idx: int = 0,
        source: str = "xt_from_vw",
        **plot_kwargs,
    ):


        decodertype = str(_get_attr(self.prs, "decodertype", "lineardecoder"))
        readout = self.stats.get(decodertype)
        if readout is None:
            raise ValueError("No decoder results found. Run regress_popreadout() first.")
        return plot_one_ff_decoding.plot_decoded_trajectory(
            readout,
            trial_idx=trial_idx,
            source=source,
            **plot_kwargs,
        )

    def plot_decoder_correlation_bars(
        self,
        *,
        varnames: Optional[Sequence[str]] = None,
        **plot_kwargs,
    ):
        decodertype = str(_get_attr(self.prs, "decodertype", "lineardecoder"))
        readout = self.stats.get(decodertype)
        if readout is None:
            raise ValueError("No decoder results found. Run regress_popreadout() first.")
        return plot_one_ff_decoding.plot_decoder_correlation_bars(
            readout,
            varnames=varnames,
            **plot_kwargs,
        )


    def plot_all_decoding_results(
        self,
        *,
        parity_varnames: Optional[Sequence[str]] = None,
        bar_varnames: Optional[Sequence[str]] = None,
        trial_idx: int = 0,
        traj_source: str = "xt_from_vw",
    ):

        canoncorr = self.stats.get("trialtype", {}).get("all", {}).get("canoncorr")
        decodertype = str(_get_attr(self.prs, "decodertype", "lineardecoder"))
        readout = self.stats.get(decodertype)
        return plot_one_ff_decoding.plot_all_decoding_results(
            canoncorr_block=canoncorr,
            readout_block=readout,
            parity_varnames=parity_varnames,
            bar_varnames=bar_varnames,
            trial_idx=trial_idx,
            traj_source=traj_source,
        )

    def plot_single_trial_decoding_panel(
        self,
        *,
        trial_indices: Optional[Sequence[int]] = None,
        n_trials: int = 6,
        **plot_kwargs,
    ):
        decodertype = str(_get_attr(self.prs, "decodertype", "lineardecoder"))
        readout = self.stats.get(decodertype)
        if readout is None:
            raise ValueError("No decoder results found. Run regress_popreadout() first.")
        return plot_one_ff_decoding.plot_single_trial_decoding_panel(
            readout,
            trial_indices=trial_indices,
            n_trials=n_trials,
            **plot_kwargs,
        )

    def run(
        self,
        *,
        save_dir: Optional[str] = None,
        cv_mode: Optional[str] = None,
    ) -> Dict:
        canoncorr_path = (
            str(Path(save_dir) / "canoncorr.pkl") if save_dir else None
        )
        readout_path = (
            str(Path(save_dir) / "lineardecoder.pkl") if save_dir else None
        )
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if bool(_get_attr(self.prs, "compute_canoncorr", True)):
            self.compute_canoncorr(save_path=canoncorr_path)
        if bool(_get_attr(self.prs, "regress_popreadout", True)):
            self.regress_popreadout(save_path=readout_path, cv_mode=cv_mode)
        return self.stats


def run_one_ff_population_decoding(
    *,
    session_num: int = 0,
    prs=None,
    mat_path: str = "all_monkey_data/one_ff_data/sessions_python.mat",
    output_root: str = "all_monkey_data/one_ff_data/decoding",
) -> Dict:
    runner = OneFFDecodingRunner(
        session_num=session_num,
        prs=prs,
        mat_path=mat_path,
        output_root=output_root,
    )
    return runner.run()
