# =========================
# For all points: from monkey_information
# =========================

global_time = [
    'time',
    'cum_distance',
]

core_kinematics = [
    'monkey_x', 'monkey_y', 'monkey_angle',
    'speed', 'accel',
    'ang_speed', 'ang_accel',
    # 'curv_of_traj',
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
    'stop', # can use spline instead of dummy
    'crossing_boundary', # can use spline instead of dummy
    'delta_distance',
]


# =========================
# For all points: Additional features to get
# =========================

ff_visibility = [
    'num_ff_visible', 'log1p_num_ff_visible',
    'num_ff_in_memory', 'log1p_num_ff_in_memory',
    'capture_ff'
]


target_related = [
    'target_rel_x', 'target_rel_y', 
    'target_angle', 'target_distance',
    'target_angle_to_boundary',
    'target_vis', 
    'time_since_target_last_seen',
    'cum_distance_since_target_last_seen',
    'd_heading_since_target_last_seen',
    'time_since_last_capture',
    'target_last_seen_distance',
    'target_last_seen_angle',
]


best_arc_vis_ff_related = [
    'best_arc_vis_ff_rel_x', 'best_arc_vis_ff_rel_y',
    'best_arc_vis_ff_distance',
    'best_arc_vis_ff_angle', 'best_arc_vis_ff_angle_to_boundary',
    'best_arc_vis_ff_vis',
]

# =========================
# B. Only in glm stop design
# =========================

stop_related = [
    'stop_id_duration',
]

# =========================
# C. Only in pn glm design
# =========================

cur_and_nxt_ff_visibility = [
    #'cur_ff_index', 'nxt_ff_index', 
    'cur_vis', 'nxt_vis',
    'cur_in_memory', 'nxt_in_memory',
]

firefly_geometry = [
    'cur_ff_distance', 'nxt_ff_distance',
    'cur_ff_angle', 'nxt_ff_angle',
    'cur_ff_rel_x', 'cur_ff_rel_y',
    'nxt_ff_rel_x', 'nxt_ff_rel_y',
]

firefly_geometry_abs = [
    'abs_cur_ff_angle', 'abs_nxt_ff_angle',
    'abs_cur_ff_rel_x', 'abs_nxt_ff_rel_x',
]

firefly_relations = [
    'angle_from_cur_ff_to_nxt_ff',
    'angle_opt_cur_end_to_nxt_ff',
]

firefly_relations_abs = [
    'abs_angle_from_cur_ff_to_nxt_ff',
    'abs_angle_opt_cur_end_to_nxt_ff',
]

angle_differences = [
    'diff_in_angle_to_nxt_ff',
    'diff_in_abs_angle_to_nxt_ff',
    'abs_diff_in_angle_to_nxt_ff',
    'abs_diff_in_abs_angle_to_nxt_ff'
]

curvature = [
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


stop_clusters = [
    'stop_cluster_size',
]



