

all_features = ['cur_in_memory', 'nxt_in_memory', 'cur_vis', 'nxt_vis',
                'cur_ff_distance', 'nxt_ff_distance', 'cur_ff_angle', 'nxt_ff_angle',
                'abs_cur_ff_angle', 'abs_nxt_ff_angle',
                'abs_cur_ff_rel_x', 'abs_nxt_ff_rel_x',
                'stop', 'speed', 'angular_speed', 'curv_of_traj', 'capture_ff',
                'whether_test', 'turning_right', 'time_since_last_capture',
                'accel', 'ang_accel', 'target_cluster_last_seen_distance',
                'target_cluster_last_seen_angle_to_boundary',
                'num_ff_visible', 'num_ff_in_memory', 'log1p_num_ff_visible', 'log1p_num_ff_in_memory',

                # features added after inspecting loadings (mostly in the 1st dim):
                'cum_distance_since_target_last_seen', 'time_rel_to_stop',
                'target_cluster_has_disappeared_for_last_time_dummy',
                'opt_curv_to_cur_ff', 'cur_cntr_arc_curv',
                'diff_in_angle_to_nxt_ff', 'diff_in_abs_angle_to_nxt_ff',
                ]


temporal_vars = ['capture_ff', 'log1p_num_ff_visible', 'log1p_num_ff_in_memory', 'cluster_around_target',
                 'cur_in_memory', 'nxt_in_memory', 'cur_vis', 'nxt_vis', 'target_cluster_has_disappeared_for_last_time_dummy']


def select_features(data):

    data_sub = data[all_features].copy()

    return data_sub


# Note that we have renamed variables to make them easier to understand. E.g.
    # data = data.rename(columns={'monkey_speeddummy': 'stop',
    #                             'ang_speed': 'angular_speed',
    #                             })
