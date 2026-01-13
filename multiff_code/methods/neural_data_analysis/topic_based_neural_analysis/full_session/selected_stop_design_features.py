# =========================
# Pruned GLM predictor set
# =========================

stop_design_predictors = [ 
    # --- Event-aligned temporal kernel ---
    'prepost',
    'rcos_-0.24s',
    'rcos_-0.16s',
    'rcos_-0.08s',
    'rcos_+0.00s',
    'rcos_+0.08s',
    'rcos_+0.16s',
    'rcos_+0.24s',

    # --- Core interactions ---
    'prepost*speed',

    # --- Event identity ---
    'captured',

    # --- Event × temporal kernel ---
    'rcos_-0.24s*captured',
    'rcos_-0.16s*captured',
    'rcos_-0.08s*captured',
    'rcos_+0.00s*captured',
    'rcos_+0.08s*captured',
    'rcos_+0.16s*captured',
    'rcos_+0.24s*captured',

    # --- Cluster / episode structure (slow variables) ---
    'prev_gap_s_z',
    'next_gap_s_z',
    'cluster_duration_s_z',
    'cluster_progress_c',
    'cluster_progress_c2',
    'event_is_first_in_cluster',


    # --- Policy / outcome state ---
    'rsw_first',
    'rcap_first',
    'rsw_middle',
    'rcap_middle',
    'rsw_last',
    'rcap_last',
    'one_stop_miss',
    'whether_in_retry_series',
    'miss',

    # =========================
    # Retry modulation (kept)
    # =========================

    # --- Retry × kinematics ---
    'accel*retry',
    'speed*retry',
    'ang_speed*retry',

    # --- Retry × event kernel ---
    'prepost*retry',
    'rcos_-0.24s*retry',
    'rcos_-0.16s*retry',
    'rcos_-0.08s*retry',
    'rcos_+0.00s*retry',
    'rcos_+0.08s*retry',
    'rcos_+0.16s*retry',
    'rcos_+0.24s*retry',

    # --- Retry × belief ---
    'num_ff_visible*retry',
    'num_ff_in_memory*retry',

    # --- Retry × episode progress ---
    'cluster_progress_c*retry',
    'cluster_progress_c2*retry',

]


additional_stop_design_predictors = [    # --- Kinematics ---
    'accel',
    'speed',
    'ang_speed',

    # --- Belief state (single encoding) ---
    'num_ff_visible',
    'num_ff_in_memory', ]