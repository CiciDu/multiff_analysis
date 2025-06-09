# Core trial information
trial_info_columns = [
    'bin',
    'point_index',
]

# Monkey movement and position features
monkey_movement_columns = [
    'monkey_speed',
    'monkey_angle',
    'monkey_dw',
    'monkey_ddw',
    'monkey_ddv',
    'monkey_speeddummy',
    'whether_new_distinct_stop',
    'delta_distance',
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
    'any_ff_visible',
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
    #'target_visible_dummy', # this has 0 for all cells in decoding targets
    'time_since_last_capture',
    'traj_curv',
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
extra_columns_for_concat_trials = [
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
extra_columns_for_aligned_trials = []
