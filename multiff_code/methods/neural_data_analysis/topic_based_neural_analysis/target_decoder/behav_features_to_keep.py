# Core trial information
trial_info_columns = [
    'bin',
    'point_index',
]

# Monkey movement and position features
monkey_movement_columns = [
    'speed',
    'monkey_angle',
    'ang_speed',
    'ang_accel',
    'accel',
    'stop_time_ratio_in_bin',
    'whether_new_distinct_stop',
    'delta_distance',
    'monkey_x_target_last_seen',
    'monkey_y_target_last_seen',
]

# Eye tracking and gaze features
eye_tracking_columns = [
    'LDy', 'LDz', 'RDy', 'RDz',
    'gaze_mky_view_x_l', 'gaze_mky_view_y_l', 'gaze_mky_view_angle_l',
    'gaze_mky_view_x_r', 'gaze_mky_view_y_r', 'gaze_mky_view_angle_r',
    'eye_world_speed',
    'valid_view_point_l', 'valid_view_point_r',
]

# Firefly-related features
firefly_columns = [
    'num_alive_ff',
    'num_visible_ff',
    'min_ff_distance',
    'min_abs_ff_angle',
    'min_abs_ff_angle_boundary',
    'min_visible_ff_distance',
    'min_abs_visible_ff_angle',
    'min_abs_visible_ff_angle_boundary',
    'catching_ff',
    'log1p_num_ff_visible',
]

# Target-related features
target_columns = [
    'target_distance',
    'target_angle',
    'target_angle_to_boundary',
    'target_rel_x',
    'target_rel_y',
    'time_since_target_last_seen',
    'target_last_seen_distance',
    'target_last_seen_angle',
    'target_last_seen_angle_to_boundary',
    # 'target_visible_dummy', # this has 0 for all cells in decoding targets
    'time_since_last_capture',
    'curv_of_traj',
    'target_opt_arc_dheading',
    'time_target_last_seen',
    'distance_from_monkey_pos_target_last_seen',
    'cum_distance_since_target_last_seen',
    'd_heading_since_target_last_seen'
]

# Combine all shared columns
shared_columns_to_keep = (
    trial_info_columns +
    monkey_movement_columns +
    eye_tracking_columns +
    firefly_columns +
    target_columns
)

# Additional columns for stitched time data
retries_columns_for_concat_trials = [
    'time',
    'monkey_x',
    'monkey_y',
    'cum_distance',
    'gaze_world_x_l',
    'gaze_world_y_l',
    'gaze_world_x_r',
    'gaze_world_y_r',
    'target_index',
    'target_x',
    'target_y',
]

# Additional columns for aligned trials
retries_columns_for_aligned_trials = []


behav_features_to_drop = [
    'bin_end_time',
    'bin_start_time',
    'crossing_boundary',
    'cum_distance_when_target_last_seen',
    'current_target_caught_time',
    'dt',
    'gaze_mky_view_angle',
    'gaze_mky_view_x',
    'gaze_mky_view_y',
    'gaze_world_x',
    'gaze_world_y',
    'last_target_caught_time',
    'monkey_angle_target_last_seen',
    'target_has_disappeared_for_last_time_dummy',
    'target_visible_dummy',
    'capture_ff',
    'trial',
    'turning_right',
    'valid_view_point'
]
