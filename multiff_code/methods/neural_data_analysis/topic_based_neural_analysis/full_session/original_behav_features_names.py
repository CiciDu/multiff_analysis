# =========================
# A. Shared variables
# =========================

global_time = [
    'time',
    'point_index',
]

core_kinematics = [
    'monkey_x', 'monkey_y', 'monkey_angle',
    'speed', 'accel',
    'ang_speed', 'ang_accel',
]

motion_signals = [
    'LDy', 'LDz', 'RDy', 'RDz'
]

gaze = [
    'gaze_mky_view_x', 'gaze_mky_view_y', 'gaze_mky_view_angle',
    'gaze_world_x', 'gaze_world_y',

    'gaze_mky_view_x_l', 'gaze_mky_view_y_l', 'gaze_mky_view_angle_l',
    'gaze_world_x_l', 'gaze_world_y_l',

    'gaze_mky_view_x_r', 'gaze_mky_view_y_r', 'gaze_mky_view_angle_r',
    'gaze_world_x_r', 'gaze_world_y_r'
]

derived_motion = [
    'eye_world_speed',
    'monkey_speeddummy',
    'crossing_boundary',
    'delta_distance',
    'cum_distance',
    'dt'
]

# stop_related = [
#     'stop_time',
#     'stop_id_duration',
#     'stop_id_start_time',
#     'stop_id_end_time',
#     'turning_right',
#     'trial'
# ]

# =========================
# B. Only in glm stop design
# =========================

# segmentation_minimal = [
#     'new_segment',
#     'new_bin',
#     'new_seg_start_time',
#     'new_seg_end_time',
#     'new_seg_duration'
# ]


# =========================
# C. Only in pn glm design
# =========================

# segment_stop_timing = [
#     'segment', 'segment_duration',
#     'seg_start_time', 'seg_end_time',
#     'stop_time',
#     'stop_point_index',
#     'next_stop_point_index',
#     'next_stop_time',
#     'time_rel_to_stop'
# ]

firefly_identity_visibility = [
    #'cur_ff_index', 'nxt_ff_index', 
    'target_index',
    'cur_vis', 'nxt_vis',
    'cur_in_memory', 'nxt_in_memory',
    'num_ff_visible', 'log1p_num_ff_visible',
    'num_ff_in_memory', 'log1p_num_ff_in_memory',
    'capture_ff'
]

firefly_geometry = [
    'cur_ff_distance', 'nxt_ff_distance',
    'cur_ff_angle', 'nxt_ff_angle',
    'cur_ff_rel_x', 'cur_ff_rel_y',
    'nxt_ff_rel_x', 'nxt_ff_rel_y',
    'cur_ff_distance_at_ref', 'nxt_ff_distance_at_ref',
    'cur_ff_angle_at_ref', 'nxt_ff_angle_at_ref'
]

firefly_geometry_abs = [
    'abs_cur_ff_angle', 'abs_nxt_ff_angle',
    'abs_cur_ff_rel_x', 'abs_nxt_ff_rel_x',
    'abs_cur_ff_angle_at_ref', 'abs_nxt_ff_angle_at_ref'
]

firefly_relations = [
    'angle_from_cur_ff_to_nxt_ff',
    'dir_from_cur_ff_to_nxt_ff',
    'angle_from_cur_ff_to_stop',
    'dir_from_cur_ff_to_stop',
    'angle_from_stop_to_nxt_ff',
    'angle_opt_cur_end_to_nxt_ff',
    'angle_from_m_before_stop_to_cur_ff'
]

firefly_relations_abs = [
    'abs_angle_from_cur_ff_to_nxt_ff',
    'abs_angle_from_cur_ff_to_stop',
    'abs_angle_from_stop_to_nxt_ff',
    'abs_angle_opt_cur_end_to_nxt_ff',
    'abs_angle_from_m_before_stop_to_cur_ff'
]

angle_differences = [
    'diff_in_angle_to_nxt_ff',
    'diff_in_abs_angle_to_nxt_ff',
    'abs_diff_in_angle_to_nxt_ff',
    'abs_diff_in_abs_angle_to_nxt_ff'
]

curvature = [
    'curv_of_traj',
    'traj_curv_to_stop',
    'curv_from_stop_to_nxt_ff',
    'curv_from_cur_end_to_nxt_ff',
    'opt_curv_to_cur_ff'
]

curvature_optimal = [
    'cur_opt_arc_curv',
    'cur_cntr_arc_curv',
    'nxt_opt_arc_curv',
    'nxt_cntr_arc_curv',
    'cur_opt_arc_dheading',
    'nxt_opt_arc_dheading',
    'cur_opt_arc_end_heading'
]

curvature_residuals = [
    'd_curv_null_arc', 'd_curv_monkey',
    'abs_d_curv_null_arc', 'abs_d_curv_monkey',
    'diff_in_d_curv', 'diff_in_abs_d_curv'
]

curvature_summary = [
    'curv_iqr', 'curv_range'
]

eye_raw = [
    'cur_eye_hor_l', 'cur_eye_ver_l',
    'cur_eye_hor_r', 'cur_eye_ver_r',
    'nxt_eye_hor_l', 'nxt_eye_ver_l',
    'nxt_eye_hor_r', 'nxt_eye_ver_r'
]

target_basic = [
    'target_x', 'target_y', 
    'target_rel_x', 'target_rel_y',
    'target_angle', 'target_distance',
    'target_visible_dummy'
]

target_memory = [
    'time_target_last_seen',
    'time_since_target_last_seen',
    'time_since_last_capture',
    'target_last_seen_distance',
    'target_last_seen_angle',
    'distance_from_monkey_pos_target_last_seen'
]

target_boundary_disappearance = [
    'target_angle_to_boundary',
    'target_last_seen_angle_to_boundary',
    'target_cluster_last_seen_angle',
    'target_cluster_last_seen_angle_to_boundary',
    'target_cluster_last_seen_distance',
    'target_cluster_last_seen_time',
    'target_cluster_visible_dummy',
    'target_cluster_has_disappeared_for_last_time_dummy',
    'target_has_disappeared_for_last_time_dummy'
]

stop_clusters = [
    'stop_cluster_id',
    'stop_cluster_size',
    'stop_cluster_start_point',
    'stop_cluster_end_point',
    'whether_new_distinct_stop'
]

monkey_state_at_last_seen = [
    'monkey_x_target_last_seen',
    'monkey_y_target_last_seen',
    'monkey_angle_target_last_seen',
    'monkey_x_target_cluster_last_seen',
    'monkey_y_target_cluster_last_seen',
    'monkey_angle_target_cluster_last_seen'
]

distance_heading_since_memory = [
    'cum_distance_since_target_last_seen',
    'cum_distance_when_target_last_seen',
    'cum_distance_target_cluster_last_seen',
    'd_heading_since_target_last_seen'
]

validity_flags = [
    'valid_view_point',
    'valid_view_point_l',
    'valid_view_point_r',
    'whether_test',
]

