eye_position_features = [
                     'LDy_std',
                     'LDz_std',
                     'RDy_std',
                     'RDz_std',
                     'LDy_range',
                     'LDy_iqr',
                     'LDz_range',
                     'LDz_iqr',
                     'RDy_range',
                     'RDy_iqr',
                     'RDz_range',
                     'RDz_iqr']

eye_pos_rel_to_cur_ff_features = ['left_eye_cur_ff_time_perc_5',
                                  'left_eye_cur_ff_time_perc_10',
                                  'right_eye_cur_ff_time_perc_5',
                                  'right_eye_cur_ff_time_perc_10']

eye_pos_rel_to_nxt_ff_features = ['left_eye_nxt_ff_time_perc_5',
                                  'left_eye_nxt_ff_time_perc_10',
                                  'right_eye_nxt_ff_time_perc_5',
                                  'right_eye_nxt_ff_time_perc_10']


all_eye_features = eye_position_features + eye_pos_rel_to_cur_ff_features + eye_pos_rel_to_nxt_ff_features

trajectory_features = ['monkey_speed_range',
                       'monkey_speed_iqr',
                       'monkey_dw_range',
                       'monkey_dw_iqr',
                       'monkey_speed_std',
                       'monkey_dw_std',
                       'curv_range',
                       'curv_iqr',
                       'curv_mean',
                       'curv_std',
                       'curv_min',
                       'curv_Q1',
                       'curv_median',
                       'curv_Q3',
                       ]

traj_to_cur_ff_features = ['d_heading_of_traj',
                            'ref_curv_of_traj',
                            'curv_of_traj_before_stop']

    
planning_indicators = ['diff_in_d_heading_to_cur_ff',
                           'dir_from_cur_ff_to_stop',
                           'diff_in_angle_to_nxt_ff',
                           'diff_in_abs_angle_to_nxt_ff',
                           ]

cur_ff_at_ref_features = ['cur_ff_distance_at_ref',
                           'cur_ff_angle_at_ref',
                           'cur_ff_angle_boundary_at_ref',
                           'cur_ff_angle_diff_boundary_at_ref']

nxt_ff_at_ref_features = ['nxt_ff_distance_at_ref',
                           'nxt_ff_angle_at_ref',
                           'nxt_ff_angle_boundary_at_ref',
                           'nxt_ff_angle_diff_boundary_at_ref']


