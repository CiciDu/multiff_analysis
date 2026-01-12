# =========================================
# MultiFF GLM predictors (final organization)
# =========================================

# -------------------------
# 1. Intercept
# -------------------------
# intercept = [
#     'const',
# ]

# -------------------------
# 2. Visibility & memory state (current / next)
# -------------------------
visibility_state = [
    'cur_vis',
    'nxt_vis',
    'cur_in_memory',
    'nxt_in_memory',
]

# -------------------------
# 3. Visibility transition dynamics (on / off transients)
# -------------------------
visibility_transients = [
    'cur_vis_on:b0:0',
    'cur_vis_on:b0:1',
    'cur_vis_on:b0:2',
    'cur_vis_off:b0:0',
    'cur_vis_off:b0:1',
    'cur_vis_off:b0:2',
]

# -------------------------
# 4. Kinematics (movement state)
# -------------------------
kinematics = [
    'speed:s0',
    'speed:s1',
    'speed:s2',
    'ang_speed:s0',
    'ang_speed:s1',
    'ang_speed:s2',
    'curv_of_traj',
]

# -------------------------
# 5. Firefly count & memory load
# -------------------------
ff_count_memory = [
    'num_ff_visible',
    'log1p_num_ff_visible',
    'log1p_num_ff_in_memory',
]

# -------------------------
# 6. Firefly egocentric geometry
# -------------------------
ff_angle = [
    'cur_ff_angle:sin1',
    'cur_ff_angle:cos1',
    'nxt_ff_angle:sin1',
    'nxt_ff_angle:cos1',
]

ff_distance = [
    'cur_ff_distance_spline:s0',
    'cur_ff_distance_spline:s1',
    'cur_ff_distance_spline:s2',
    'cur_ff_distance_spline:s3',
    'cur_ff_distance_spline:s4',
    'nxt_ff_distance_spline:s0',
    'nxt_ff_distance_spline:s1',
    'nxt_ff_distance_spline:s2',
    'nxt_ff_distance_spline:s3',
    'nxt_ff_distance_spline:s4',
]

# -------------------------
# 7. Temporal memory & experience
# -------------------------
temporal_context = [
    'time_since_target_last_seen',
    'time_since_last_capture',
    'cum_dist_seen_log1p',
]









# -------------------------
# 8. Gaze state (control)
# -------------------------
gaze_state = [
    'eye_speed_log1p',
    'gaze_mky_view_x_z',
    'gaze_mky_view_y_z',
    'gaze_mky_view_x_z*gaze_mky_view_y_z',
]

# -------------------------
# 9. Eye motion energy (left / right eye)
# -------------------------
eye_motion_energy = [
    'RDz_odd',
    'RDz_mag',
    'RDy_odd',
    'RDy_mag',
    'LDz_odd',
    'LDz_mag',
    'LDy_odd',
    'LDy_mag',
]

# -------------------------
# 10. Acceleration & angular acceleration
# -------------------------
acceleration_state = [
    'accel_odd',
    'accel_mag',
    'ang_accel_odd',
    'ang_accel_mag',
]

# -------------------------
# Full ordered predictor list
# -------------------------
pn_design_predictors = (
    visibility_state
    + visibility_transients
    + kinematics
    + ff_count_memory
    + ff_angle
    + ff_distance
    + temporal_context
    # + gaze_state
    # + eye_motion_energy
    # + acceleration_state
)
