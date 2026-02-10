import numpy as np
import pandas as pd
from neural_data_analysis.design_kits.design_by_segment import temporal_feats
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import (
    one_ff_glm_design,
    one_ff_pipeline,
    parameters
)
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import (
    one_ff_gam_fit
)


def build_tuning_design(
    data_df,
    linear_vars,
    angular_vars,
    n_bins=10,
    binrange_dict=None,
):
    """
    Build continuous tuning design (no unit dependence).

    Parameters
    ----------
    data_df : DataFrame
        Data with covariates
    linear_vars : list
        Linear variable names
    angular_vars : list
        Angular variable names
    n_bins : int
        Number of bins (default: 10)
    binrange_dict : dict, optional
        Dictionary mapping variable names to [min, max] ranges from parameters

    Returns
    -------
    X_tuning : DataFrame
        Tuning design matrix
    tuning_meta : dict
        Metadata
    """
    X_tuning, tuning_meta = one_ff_glm_design.build_continuous_tuning_block(
        data=data_df,
        linear_vars=linear_vars,
        angular_vars=angular_vars,
        n_bins=n_bins,
        center=True,
        binrange_dict=binrange_dict,
    )

    return X_tuning, tuning_meta


def build_temporal_design_base(
    data_obj,
):
    """
    Build temporal design matrix that is reusable across units.
    Spike history and coupling are NOT included here.
    Uses binrange from data_obj.prs for temporal filter bounds.
    """

    specs, specs_meta = temporal_feats._init_predictor_specs(
        data_obj.prs.dt,
        data_obj.trial_ids,
    )

    # Get binrange for temporal filters if available
    binrange = data_obj.prs.binrange if hasattr(
        data_obj.prs, 'binrange') else {}

    # Store basis info for plotting (lags and basis functions)
    basis_info = {}

    # t_move: use binrange if available, otherwise default [-0.3, 0.3]
    if 't_move' in binrange:
        t_move_min = float(binrange['t_move'][0])
        t_move_max = float(binrange['t_move'][1])
    else:
        t_move_min, t_move_max = -0.3, 0.3

    lags_move, B_move = glm_bases.raised_cosine_basis(
        n_basis=10,
        t_min=t_move_min,
        t_max=t_move_max,
        dt=specs_meta['dt'],
    )
    basis_info['t_move'] = {'lags': lags_move, 'basis': B_move}

    # t_targ: use binrange if available, otherwise default [0.0, 0.6]
    if 't_targ' in binrange:
        t_targ_min = float(binrange['t_targ'][0])
        t_targ_max = float(binrange['t_targ'][1])
    else:
        t_targ_min, t_targ_max = 0.0, 0.6

    lags_targ, B_targ = glm_bases.raised_cosine_basis(
        n_basis=10,
        t_min=t_targ_min,
        t_max=t_targ_max,
        dt=specs_meta['dt'],
    )
    basis_info['t_targ'] = {'lags': lags_targ, 'basis': B_targ}

    specs['t_targ'] = temporal_feats.PredictorSpec(
        signal=data_obj.events['t_targ'],
        bases=[B_targ],
    )
    specs['t_move'] = temporal_feats.PredictorSpec(
        signal=data_obj.events['t_move'],
        bases=[B_move],
    )

    # t_rew: use binrange if available and different from t_move
    if 't_rew' in binrange and not np.array_equal(binrange.get('t_rew'), binrange.get('t_move')):
        t_rew_min = float(binrange['t_rew'][0])
        t_rew_max = float(binrange['t_rew'][1])
        lags_rew, B_rew = glm_bases.raised_cosine_basis(
            n_basis=10,
            t_min=t_rew_min,
            t_max=t_rew_max,
            dt=specs_meta['dt'],
        )
        basis_info['t_rew'] = {'lags': lags_rew, 'basis': B_rew}
    else:
        B_rew = B_move  # reuse B_move if no specific binrange
        basis_info['t_rew'] = basis_info['t_move']
    
    specs['t_rew'] = temporal_feats.PredictorSpec(
        signal=data_obj.events['t_rew'],
        bases=[B_rew],
    )
    
    # t_stop: use binrange if available and different from t_move
    if 't_stop' in binrange and not np.array_equal(binrange.get('t_stop'), binrange.get('t_move')):
        t_stop_min = float(binrange['t_stop'][0])
        t_stop_max = float(binrange['t_stop'][1])
        lags_stop, B_stop = glm_bases.raised_cosine_basis(
            n_basis=10,
            t_min=t_stop_min,
            t_max=t_stop_max,
            dt=specs_meta['dt'],
        )
        basis_info['t_stop'] = {'lags': lags_stop, 'basis': B_stop}
    else:
        B_stop = B_move  # reuse B_move if no specific binrange
        basis_info['t_stop'] = basis_info['t_move']

    specs['t_stop'] = temporal_feats.PredictorSpec(
        signal=data_obj.events['t_stop'],
        bases=[B_stop],
    )

    temporal_df, temporal_meta = temporal_feats.specs_to_design_df(
        specs,
        data_obj.covariate_trial_ids,
        edge='zero',
        add_intercept=True,
        respect_trial_boundaries=False,
    )
    
    # Add basis info to metadata
    temporal_meta['basis_info'] = basis_info

    return temporal_df, temporal_meta, specs_meta


def build_design_df(
    unit_idx,
    data_obj,
    temporal_df,
    temporal_meta,
    X_tuning,
    tuning_meta,
    specs_meta,
    coupling_units=None,
):
    """
    Assemble the full design matrix for one unit.
    """

    # ----------------------------
    # Spike history + coupling specs
    # ----------------------------
    specs = {}
    basis_info = {}

    lags_hist, B_hist = glm_bases.raised_log_cosine_basis(
        n_basis=10,
        t_min=0.0,
        t_max=0.35,
        dt=specs_meta['dt'],
        log_spaced=True,
    )
    basis_info['spike_hist'] = {'lags': lags_hist, 'basis': B_hist}

    specs['spike_hist'] = temporal_feats.PredictorSpec(
        signal=data_obj.Y[:, unit_idx],
        bases=[B_hist],
    )

    if coupling_units is not None:
        lags_coup, B_coup = glm_bases.raised_log_cosine_basis(
            n_basis=10,
            t_min=0.0,
            t_max=1.375,
            dt=specs_meta['dt'],
        )

        for j in coupling_units:
            specs[f'cpl_{j}'] = temporal_feats.PredictorSpec(
                signal=data_obj.Y[:, j],
                bases=[B_coup],
            )
            basis_info[f'cpl_{j}'] = {'lags': lags_coup, 'basis': B_coup}

    hist_df, hist_meta = temporal_feats.specs_to_design_df(
        specs,
        data_obj.covariate_trial_ids,
        edge='zero',
        add_intercept=False,
        respect_trial_boundaries=False,
    )
    
    # Add basis info to metadata
    hist_meta['basis_info'] = basis_info

    # ----------------------------
    # Concatenate all components
    # ----------------------------
    X_tuning = X_tuning.reindex(temporal_df.index)
    hist_df = hist_df.reindex(temporal_df.index)

    design_df = pd.concat(
        [temporal_df, hist_df, X_tuning],
        axis=1,
    )

    # Apply valid row mask if present
    rows_mask = temporal_meta.get('valid_rows_mask', None)
    if rows_mask is not None:
        design_df = design_df.loc[rows_mask]

    return design_df, hist_meta


def extract_response(
    unit_idx,
    data_obj,
    design_df,
    temporal_meta,
):
    """
    Extract spike count response aligned to design_df.
    """
    y = pd.Series(
        data_obj.Y[:, unit_idx],
        index=design_df.index,
    ).to_numpy()

    rows_mask = temporal_meta.get('valid_rows_mask', None)
    if rows_mask is not None:
        y = y[rows_mask]

    return y


def build_group_specs(
    temporal_meta,
    tuning_meta,
    hist_meta,
    lam_f=100.0,  # firefly features (tuning curves)
    lam_g=10.0,  # temporal event kernels (t_*)
    lam_h=10.0,  # spike history
    lam_p=10.0,  # coupling
    coupling_units=None,
):
    """
    Construct GroupSpec list for GAM fitting.
    """
    groups = []

    # ----------------------------
    # Event kernels (temporal)
    # ----------------------------
    tg = temporal_meta['groups']
    groups.extend([
        one_ff_gam_fit.GroupSpec('t_targ', tg['t_targ'], 'event', lam_g),
        one_ff_gam_fit.GroupSpec('t_move', tg['t_move'], 'event', lam_g),
        one_ff_gam_fit.GroupSpec('t_rew',  tg['t_rew'],  'event', lam_g),
        one_ff_gam_fit.GroupSpec('t_stop', tg['t_stop'], 'event', lam_g),
    ])

    # ----------------------------
    # Spike history
    # ----------------------------
    hg = hist_meta['groups']
    groups.append(one_ff_gam_fit.GroupSpec(
        'spike_hist',
        hg['spike_hist'],
        'event',
        lam_h,
    ))

    # ----------------------------
    # Coupling
    # ----------------------------
    if coupling_units is not None:
        for j in coupling_units:
            groups.append(one_ff_gam_fit.GroupSpec(
                f'cpl_{j}',
                hg[f'cpl_{j}'],
                'event',
                lam_p,
            ))

    # ----------------------------
    # Tuning curves (ALL 1D, paper-faithful)
    # ----------------------------
    for var, cols in tuning_meta['groups'].items():
        groups.append(one_ff_gam_fit.GroupSpec(
            var,
            cols,
            '1D',
            lam_f,
        ))

    return groups


def process_unit_design_and_groups(
    unit_idx,
    data_obj,
    temporal_df,
    temporal_meta,
    X_tuning,
    tuning_meta,
    specs_meta,
    lam_f=100.0,
    lam_g=10.0,
    lam_h=10.0,
    lam_p=10.0,
    coupling_units=None,
):
    """
    Assemble full design matrix, response, and GroupSpecs for one unit.

    Returns
    -------
    design_df : DataFrame
        Full design matrix
    y : ndarray
        Response variable
    groups : List[GroupSpec]
        Group specifications
    all_meta : dict
        Combined metadata with 'tuning', 'temporal', and 'hist' sub-dicts
    """

    design_df, hist_meta = build_design_df(
        unit_idx=unit_idx,
        data_obj=data_obj,
        temporal_df=temporal_df,
        temporal_meta=temporal_meta,
        X_tuning=X_tuning,
        tuning_meta=tuning_meta,
        specs_meta=specs_meta,
        coupling_units=coupling_units,
    )

    y = extract_response(
        unit_idx=unit_idx,
        data_obj=data_obj,
        design_df=design_df,
        temporal_meta=temporal_meta,
    )

    groups = build_group_specs(
        temporal_meta=temporal_meta,
        tuning_meta=tuning_meta,
        hist_meta=hist_meta,
        lam_f=lam_f,
        lam_g=lam_g,
        lam_h=lam_h,
        lam_p=lam_p,
        coupling_units=coupling_units,
    )

    # Combine all metadata for easy access
    all_meta = {
        'tuning': tuning_meta,
        'temporal': temporal_meta,
        'hist': hist_meta,
    }

    return design_df, y, groups, all_meta


def finalize_one_ff_pgam_design(unit_idx, session_num=0):
    # -------------------------------
    # Covariates
    # -------------------------------
    covariate_names = [
        'v', 'w', 'd', 'phi',
        'r_targ', 'theta_targ',
        'eye_ver', 'eye_hor',
    ]

    linear_vars = [
        'v', 'w', 'd', 'r_targ',
        'eye_ver', 'eye_hor',
    ]

    angular_vars = [
        'phi', 'theta_targ',
    ]

    # -------------------------------
    # Load data
    # -------------------------------
    prs = parameters.default_prs()

    data_obj = one_ff_pipeline.OneFFSessionData(
        mat_path='all_monkey_data/one_ff_data/sessions_python.mat',
        prs=prs,
        session_num=session_num,
    )

    # -------------------------------
    # Preprocessing
    # -------------------------------
    data_obj.compute_covariates(covariate_names)
    data_obj.compute_spike_counts()
    data_obj.smooth_spikes()
    data_obj.compute_events()

    # -------------------------------
    # Build shared design
    # -------------------------------
    temporal_df, temporal_meta, specs_meta = (
        build_temporal_design_base(data_obj)
    )

    X_tuning, tuning_meta = (
        build_tuning_design(
            data_obj.data_df,
            linear_vars,
            angular_vars,
            binrange_dict=prs.binrange,
        )
    )

    # -------------------------------
    # Per-unit GAM design
    # -------------------------------
    design_df, y, groups, all_meta = (
        process_unit_design_and_groups(
            unit_idx=unit_idx,
            data_obj=data_obj,
            temporal_df=temporal_df,
            temporal_meta=temporal_meta,
            X_tuning=X_tuning,
            tuning_meta=tuning_meta,
            specs_meta=specs_meta,
        )
    )

    return design_df, y, groups, all_meta
