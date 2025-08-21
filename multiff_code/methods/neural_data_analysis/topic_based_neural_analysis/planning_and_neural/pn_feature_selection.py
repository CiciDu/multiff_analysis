def select_features(data):
    data = data.rename(columns={'cur_ff_in_memory_dummy': 'cur_in_memory', 
                            'nxt_ff_in_memory_dummy': 'nxt_in_memory',
                            'cur_ff_visible_dummy': 'cur_vis',
                            'nxt_ff_visible_dummy': 'nxt_vis',
                            'monkey_speeddummy': 'stop',
                            'monkey_speed': 'speed',
                            'monkey_dw': 'angular_speed',
                            'curv_of_traj': 'curvature',
                            'capture_target_dummy': 'capture',
                            })


    features = ['cur_in_memory', 'nxt_in_memory', 'cur_vis', 'nxt_vis', 
                'cur_ff_distance', 'nxt_ff_distance', 'cur_ff_angle', 'nxt_ff_angle',
                'abs_cur_ff_angle', 'abs_nxt_ff_angle', 
                'abs_cur_ff_rel_x', 'abs_nxt_ff_rel_x',
                'stop', 'speed', 'angular_speed', 'curvature', 'capture',
                'whether_test', 'turning_right', 'time_since_last_capture',
                'monkey_ddv', 'monkey_ddw', 'target_cluster_last_seen_distance',
                'target_cluster_last_seen_angle_to_boundary', 

                # features added after inspecting loadings (mostly in the 1st dim):
                'cum_distance_since_target_last_seen', 'time_rel_to_stop',
                'target_cluster_has_disappeared_for_last_time_dummy',
                'opt_curv_to_cur_ff', 'cur_cntr_arc_curv',
                'abs_diff_in_angle_to_nxt_ff', 'abs_diff_in_abs_angle_to_cur_ff',
                
                ]
    data_sub = data[features].copy()

    return data_sub
        
