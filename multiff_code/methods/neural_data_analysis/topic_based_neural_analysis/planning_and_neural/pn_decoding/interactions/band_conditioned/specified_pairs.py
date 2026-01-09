
CONTINUOUS_INTERACTIONS = [
    # (target_col, condition_col)

    # =========================================================
    # 1. Core movement control (highest priority)
    # =========================================================

    # Steering geometry conditioned on speed (gain scheduling)
    ('cur_ff_angle', 'speed_band'),
    ('abs_cur_ff_angle', 'speed_band'),

    # Distance-to-current-target conditioned on speed (approach urgency)
    ('log1p_cur_ff_distance', 'speed_band'),

    # Motor output conditioned on speed (policy regime)
    ('ang_speed', 'speed_band'),
    ('accel', 'speed_band'),

    # ---------------------------------------------------------
    # NEW: Next-target distance readout under speed
    # (planning load during fast vs slow motion)
    # ---------------------------------------------------------
    ('log1p_nxt_ff_distance', 'speed_band'),

    # =========================================================
    # 2. Geometry × distance (navigation state)
    # =========================================================

    # Steering geometry conditioned on current distance (near vs far control)
    ('cur_ff_angle', 'cur_ff_dist_band'),
    ('abs_cur_ff_angle', 'cur_ff_dist_band'),

    # Lateral deviation conditioned on current distance (path correction)
    ('cur_ff_rel_x', 'cur_ff_dist_band'),
    ('cur_ff_rel_x', 'log1p_cur_ff_distance_band'),

    # ---------------------------------------------------------
    # NEW: Lateral deviation conditioned on next distance
    # (pre-alignment toward upcoming target)
    # ---------------------------------------------------------
    ('cur_ff_rel_x', 'nxt_ff_dist_band'),

    # =========================================================
    # 3. Planning / lookahead geometry
    # =========================================================

    # Current steering conditioned on next-target geometry (competition)
    ('cur_ff_angle', 'nxt_ff_angle_band'),
    ('abs_cur_ff_angle', 'nxt_ff_angle_band'),

    # Current distance conditioned on next-target distance (commit vs switch)
    ('cur_ff_distance', 'nxt_ff_dist_band'),
    ('log1p_cur_ff_distance', 'nxt_ff_dist_band'),

    # ---------------------------------------------------------
    # NEW: Next distance decoded directly, conditioned on current distance
    # (explicit planning / lookahead signal)
    # ---------------------------------------------------------
    ('log1p_nxt_ff_distance', 'cur_ff_dist_band'),

    # =========================================================
    # 4. Control dynamics / replanning
    # =========================================================

    # Steering geometry conditioned on acceleration (online correction)
    ('cur_ff_angle', 'accel_band'),
    ('abs_cur_ff_angle', 'accel_band'),

    # Distance-to-current-target conditioned on acceleration
    ('log1p_cur_ff_distance', 'accel_band'),

    # ---------------------------------------------------------
    # NEW: Next-target distance conditioned on acceleration
    # (replanning / hesitation signal)
    # ---------------------------------------------------------
    ('log1p_nxt_ff_distance', 'accel_band'),
]



DISCRETE_INTERACTIONS = [

    # =========================================================
    # 1. Core movement control (highest priority)
    # =========================================================

    # Speed × geometry: classic steering regimes
    ('speed_band', 'cur_ff_angle_band', 'speed_angle_state'),
    ('speed_band', 'cur_ff_dist_band', 'speed_distance_state'),

    # Speed × motor output
    ('speed_band', 'ang_speed_band', 'speed_turnrate_state'),
    ('speed_band', 'accel_band', 'speed_accel_state'),

    # =========================================================
    # 2. Geometry × distance (navigation state)
    # =========================================================

    # How far + how misaligned am I from current target?
    ('cur_ff_angle_band', 'cur_ff_dist_band', 'angle_distance_state'),

    # Lateral geometry vs forward progress (optional but clean)
    ('cur_ff_rel_x_band', 'cur_ff_dist_band', 'lateral_distance_state'),

    # =========================================================
    # 3. Planning / lookahead geometry
    # =========================================================

    # Current vs next target geometry (planning competition)
    ('cur_ff_angle_band', 'nxt_ff_angle_band', 'cur_next_angle_state'),

    # Commitment stage × next-target relevance
    ('cur_ff_dist_band', 'nxt_ff_dist_band', 'curdist_nextdist_state'),

    # =========================================================
    # 4. Control dynamics (policy change / replanning)
    # =========================================================

    # Acceleration conditioned on geometry
    ('accel_band', 'cur_ff_angle_band', 'accel_angle_state'),
    ('accel_band', 'cur_ff_dist_band', 'accel_distance_state'),

    # Turn acceleration vs current geometry (replanning signal)
    ('ang_accel_band', 'cur_ff_angle_band', 'angaccel_angle_state'),

    # =========================================================
    # 5. Commitment / learning (late-stage, optional)
    # =========================================================

    # Early vs late commitment interacting with geometry
    ('cur_ff_dist_ref_band', 'cur_ff_angle_band', 'commit_angle_state'),

    # Commitment timing × speed (hesitation vs execution)
    ('cur_ff_dist_ref_band', 'speed_band', 'commit_speed_state'),
]
