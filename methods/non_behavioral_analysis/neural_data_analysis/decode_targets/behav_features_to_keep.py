

shared_columns_to_keep = ['bin', 'point_index', # from monkey_information
 'LDy', 'LDz', 'RDy', 'RDz', 
'gaze_mky_view_x_l', 'gaze_mky_view_y_l', 'gaze_mky_view_angle_l', 
'gaze_mky_view_x_r', 'gaze_mky_view_y_r', 'gaze_mky_view_angle_r', 
'eye_world_speed', 'valid_view_point_l','valid_view_point_r',
'monkey_speed', 'monkey_angle', 'monkey_dw', 'monkey_ddw', 'monkey_ddv', 
'monkey_speeddummy', 'whether_new_distinct_stop',
'crossing_boundary', 'delta_distance',
'num_alive_ff', 'num_visible_ff', 'min_ff_distance',# from ff_info
'min_abs_ff_angle', 'min_abs_ff_angle_boundary',
'min_visible_ff_distance', 'min_abs_visible_ff_angle',
'min_abs_visible_ff_angle_boundary', 'catching_ff', 'any_ff_visible',
'target_distance', 'target_angle', 'target_angle_to_boundary',
'target_last_seen_time', 'target_last_seen_distance_frozen',
'target_last_seen_angle_frozen', 'target_last_seen_angle_to_boundary_frozen',
'target_visible_dummy', 'time_since_last_capture',
'traj_curv', 'target_opt_arc_dheading'
  ]


extra_columns_for_stitched_time = ['time', 'monkey_x', 'monkey_y', 'cum_distance'
 'gaze_world_x_l', 'gaze_world_y_l', 'gaze_world_x_r', 'gaze_world_y_r',
 'target_x', 'target_y']


extra_columns_for_aligned_trials = []




