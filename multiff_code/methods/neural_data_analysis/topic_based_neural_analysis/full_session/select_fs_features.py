"""
Organized regressor definitions
"""

REGRESSORS = {
    # ------------------------------------------------------------------
    # Intercept / constants
    # ------------------------------------------------------------------
    'const': [
        'const',
    ],

    # ------------------------------------------------------------------
    # Stop & capture events (binned)
    # ------------------------------------------------------------------
    'stop_events': [
        'stop:b0:0', 'stop:b0:1', 'stop:b0:2',
        'stop:b0:3', 'stop:b0:4', 'stop:b0:5',
    ],

    'capture_events': [
        'capture_ff:b0:0', 'capture_ff:b0:1', 'capture_ff:b0:2',
        'capture_ff:b0:3', 'capture_ff:b0:4', 'capture_ff:b0:5',
    ],

    # ------------------------------------------------------------------
    # Eye / gaze variables
    # ------------------------------------------------------------------
    'eye_gaze': [
        'eye_speed_log1p',
        'gaze_mky_view_x_z',
        'gaze_mky_view_y_z',
        'gaze_mky_view_x_z*gaze_mky_view_y_z',
    ],

    # ------------------------------------------------------------------
    # Retinal disparity (right / left, odd & magnitude)
    # ------------------------------------------------------------------
    'retinal_disparity': [
        'RDz_odd', 'RDz_mag', 'RDy_odd', 'RDy_mag',
        'LDz_odd', 'LDz_mag', 'LDy_odd', 'LDy_mag',
    ],

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    'kinematics': [
        'accel_odd', 'accel_mag',
        'ang_accel_odd', 'ang_accel_mag',
        'speed:s0', 'speed:s1', 'speed:s2',
        'ang_speed:s0', 'ang_speed:s1', 'ang_speed:s2',
    ],

    # ------------------------------------------------------------------
    # Firefly counts & memory
    # ------------------------------------------------------------------
    'firefly_counts': [
        'num_ff_visible',
        'log1p_num_ff_visible',
        'num_ff_in_memory',
        'log1p_num_ff_in_memory',
    ],

    # ------------------------------------------------------------------
    # Best-arc planning variables
    # ------------------------------------------------------------------
    'best_arc': [
        'best_arc_ff_angle:sin1',
        'best_arc_ff_angle:cos1',
        'best_arc_ff_distance_spline:s0',
        'best_arc_ff_distance_spline:s1',
        'best_arc_ff_distance_spline:s2',
        'best_arc_ff_distance_spline:s3',
        'best_arc_ff_distance_spline:s4',
        # 'best_arc_opt_arc_length',
        # 'best_arc_abs_curv_diff',
        # 'best_arc_opt_arc_curv:s0',
        # 'best_arc_opt_arc_curv:s1',
        # 'best_arc_opt_arc_curv:s2',
        # 'best_arc_opt_arc_curv:s3',
        # 'best_arc_opt_arc_curv:s4',
        # 'best_arc_curv_diff:s0',
        # 'best_arc_curv_diff:s1',
        # 'best_arc_curv_diff:s2',
        # 'best_arc_curv_diff:s3',
        # 'best_arc_curv_diff:s4',
        'having_best_arc_ff',
    ],

    # ------------------------------------------------------------------
    # Current vs next target state
    # ------------------------------------------------------------------
    'current_next_state': [
        'cur_vis', 
        #'cur_in_memory', 
        # 'nxt_vis', 'nxt_in_memory',
        # 'cur_vis_on:b0:0', 'cur_vis_on:b0:1', 'cur_vis_on:b0:2',
        # 'cur_vis_off:b0:0', 'cur_vis_off:b0:1', 'cur_vis_off:b0:2',
        # 'cur_ff_angle:sin1', 'cur_ff_angle:cos1',
        # 'nxt_ff_angle:sin1', 'nxt_ff_angle:cos1',
        'cur_ff_distance_spline:s0',
        'cur_ff_distance_spline:s1',
        'cur_ff_distance_spline:s2',
        'cur_ff_distance_spline:s3',
        'cur_ff_distance_spline:s4',
        # 'nxt_ff_distance_spline:s0',
        # 'nxt_ff_distance_spline:s1',
        # 'nxt_ff_distance_spline:s2',
        # 'nxt_ff_distance_spline:s3',
        # 'nxt_ff_distance_spline:s4',
    ],

    # ------------------------------------------------------------------
    # Temporal / history variables
    # ------------------------------------------------------------------
    'history': [
        'time_since_target_last_seen',
        'time_since_last_capture',
        'cum_dist_seen_log1p',
    ],

    # ------------------------------------------------------------------
    # Task structure & windows
    # ------------------------------------------------------------------
    'task_structure': [
        'bin',
        'in_pn_window',
        'prepost',
        'in_stop_window',
    ],

    # ------------------------------------------------------------------
    # Raised-cosine temporal kernels
    # ------------------------------------------------------------------
    'rcos': [
        # 'rcos_-0.24s', 'rcos_-0.16s', 'rcos_-0.08s',
        # 'rcos_+0.00s', 'rcos_+0.08s', 'rcos_+0.16s', 'rcos_+0.24s',
    ],

    # ------------------------------------------------------------------
    # Capture-related interactions
    # ------------------------------------------------------------------
    'capture_interactions': [
        'captured',
        # 'prepost*speed',
        # 'rcos_-0.24s*captured',
        # 'rcos_-0.16s*captured',
        # 'rcos_-0.08s*captured',
        # 'rcos_+0.00s*captured',
        # 'rcos_+0.08s*captured',
        # 'rcos_+0.16s*captured',
        # 'rcos_+0.24s*captured',
    ],

    # ------------------------------------------------------------------
    # Cluster / sequence structure
    # ------------------------------------------------------------------
    'clusters': [
        # 'prev_gap_s_z',
        # 'next_gap_s_z',
        'cluster_duration_s_z',
        'cluster_progress_c',
        'cluster_progress_c2',
        # 'event_is_first_in_cluster',
        'rsw_first', 'rcap_first',
        'rsw_middle', 'rcap_middle',
        'rsw_last', 'rcap_last',
    ],

    # ------------------------------------------------------------------
    # Retry / miss logic
    # ------------------------------------------------------------------
    'retry': [
        'one_stop_miss',
        'whether_in_retry_series',
        'miss',
        'accel*retry',
        'speed*retry',
        'ang_speed*retry',
        # 'prepost*retry',
        # 'rcos_-0.24s*retry',
        # 'rcos_-0.16s*retry',
        # 'rcos_-0.08s*retry',
        # 'rcos_+0.00s*retry',
        # 'rcos_+0.08s*retry',
        # 'rcos_+0.16s*retry',
        # 'rcos_+0.24s*retry',
        # 'num_ff_visible*retry',
        # 'num_ff_in_memory*retry',
        # 'cluster_progress_c*retry',
        # 'cluster_progress_c2*retry',
    ],
}

# ----------------------------------------------------------------------
# Convenience: flattened list (preserves grouping order)
# ----------------------------------------------------------------------
ALL_REGRESSORS = [
    reg for group in REGRESSORS.values() for reg in group
]
