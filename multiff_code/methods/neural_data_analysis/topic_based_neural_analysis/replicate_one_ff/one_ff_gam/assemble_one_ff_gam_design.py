import pandas as pd

from neural_data_analysis.design_kits.design_by_segment import temporal_feats
from neural_data_analysis.neural_analysis_tools.glm_tools.tpg import glm_bases
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import one_ff_glm_design
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.one_ff_gam import one_ff_gam_fit

def build_tuning_design(
    data_obj,
    linear_vars,
    angular_vars,
    n_bins=10,
):
    """
    Build continuous tuning design (no unit dependence).
    """
    data_df = pd.DataFrame(data_obj.covariates)

    X_tuning, tuning_meta = one_ff_glm_design.build_continuous_tuning_block(
        data=data_df,
        linear_vars=linear_vars,
        angular_vars=angular_vars,
        n_bins=n_bins,
        center=True,
    )

    return X_tuning, tuning_meta


def build_temporal_design_base(
    data_obj,
):
    """
    Build temporal design matrix that is reusable across units.
    Spike history and coupling are NOT included here.
    """

    specs, specs_meta = temporal_feats._init_predictor_specs(
        data_obj.prs.dt,
        data_obj.trial_ids,
    )

    _, B_move = glm_bases.raised_cosine_basis(
        n_basis=10,
        t_min=-0.3,
        t_max=0.3,
        dt=specs_meta['dt'],
    )

    _, B_targ = glm_bases.raised_cosine_basis(
        n_basis=10,
        t_min=0.0,
        t_max=0.6,
        dt=specs_meta['dt'],
    )

    specs['t_targ'] = temporal_feats.PredictorSpec(
        signal=data_obj.events['t_targ'],
        bases=[B_targ],
    )
    specs['t_move'] = temporal_feats.PredictorSpec(
        signal=data_obj.events['t_move'],
        bases=[B_move],
    )
    specs['t_rew'] = temporal_feats.PredictorSpec(
        signal=data_obj.events['t_rew'],
        bases=[B_move],
    )

    temporal_df, temporal_meta = temporal_feats.specs_to_design_df(
        specs,
        data_obj.covariate_trial_ids,
        edge='zero',
        add_intercept=True,
        respect_trial_boundaries=False,
    )

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

    _, B_hist = glm_bases.raised_log_cosine_basis(
        n_basis=10,
        t_min=0.0,
        t_max=0.35,
        dt=specs_meta['dt'],
        log_spaced=True,
    )

    specs['spike_hist'] = temporal_feats.PredictorSpec(
        signal=data_obj.Y[:, unit_idx],
        bases=[B_hist],
    )

    if coupling_units is not None:
        _, B_coup = glm_bases.raised_log_cosine_basis(
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

    hist_df, hist_meta = temporal_feats.specs_to_design_df(
        specs,
        data_obj.covariate_trial_ids,
        edge='zero',
        add_intercept=False,
        respect_trial_boundaries=False,
    )

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
    lam_f=100.0,
    lam_g=10.0,
    lam_h=10.0,
    lam_p=10.0,
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


def assemble_unit_design_and_groups(
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

    return design_df, y, groups