import pandas as pd

from neural_data_analysis.design_kits.design_around_event import (
    event_binning,
)

from neural_data_analysis.topic_based_neural_analysis.stop_event_analysis.get_stop_events import (
    get_stops_utils,
    stop_design_for_decoding,
    collect_stop_data,
    stop_design_for_encoding,
)


def assemble_stop_design_func(
    raw_data_folder_path,
    bin_width,
    global_bins_2d=None,
    for_decoding=False,
    use_encoding_design=False,
    # Optional: one_ff-style encoding design (build_stop_design_for_encoding)
    use_tuning_design=False,
    tuning_feature_mode='boxcar_only', # can be 'raw_only', 'boxcar_only', 'raw_plus_boxcar'
    binrange_dict=None,
    n_basis=20,
    t_min=-0.3,
    t_max=0.3,
    tuning_n_bins=10,
    linear_vars=None,
    angular_vars=None,
):
    
    if use_encoding_design and for_decoding:
        raise ValueError("Encoding design cannot be used with for_decoding=True")


    pn, datasets, _ = collect_stop_data.collect_stop_data_func(
        raw_data_folder_path,
        bin_width=bin_width,
    )

    captures_df, valid_captures_df, filtered_no_capture_stops_df, stops_with_stats = (
        get_stops_utils.prepare_no_capture_and_captures(
            monkey_information=pn.monkey_information,
            closest_stop_to_capture_df=pn.closest_stop_to_capture_df,
            ff_caught_T_new=pn.ff_caught_T_new,
            distance_col='distance_from_ff_to_stop',
        )
    )

    stops_with_stats['stop_time'] = stops_with_stats['stop_id_start_time']
    stops_with_stats['prev_time'] = stops_with_stats['stop_id_end_time'].shift(
        1)
    stops_with_stats['next_time'] = stops_with_stats['stop_id_start_time'].shift(
        -1)

    new_seg_info = event_binning.make_new_seg_info_for_stop_design(
        stops_with_stats,
        pn.closest_stop_to_capture_df,
        pn.monkey_information,
    )

    events_with_stats = stops_with_stats[[
        'stop_id',
        'stop_cluster_id',
        'stop_id_start_time',
        'stop_id_end_time',
    ]].rename(columns={
        'stop_id': 'event_id',
        'stop_cluster_id': 'event_cluster_id',
        'stop_id_start_time': 'event_id_start_time',
        'stop_id_end_time': 'event_id_end_time',
    })

    build_fn = (
        stop_design_for_encoding.build_stop_design_for_encoding
        if use_encoding_design
        else stop_design_for_decoding.build_stop_design
    )
    build_kw = dict(
        new_seg_info=new_seg_info,
        events_with_stats=events_with_stats,
        monkey_information=pn.monkey_information,
        spikes_df=pn.spikes_df,
        ff_dataframe=pn.ff_dataframe,
        bin_dt=bin_width,
        datasets=datasets,
        add_ff_visible_info=True,
        global_bins_2d=global_bins_2d,
        for_decoding=for_decoding,
    )
    if use_encoding_design:
        build_kw.update(
            {
                'n_basis': n_basis,
                't_min': t_min,
                't_max': t_max,
                'use_tuning_design': use_tuning_design,
                'tuning_feature_mode': tuning_feature_mode,
                'binrange_dict': binrange_dict,
                'tuning_n_bins': tuning_n_bins,
                'linear_vars': linear_vars,
                'angular_vars': angular_vars,
                'add_temporal_and_tuning_after_scale': True,
            }
        )
    build_result = build_fn(**build_kw)
    if use_encoding_design:
        (stop_binned_spikes, init_stop_binned_feats, offset_log, stop_meta_used,
         stop_meta_groups, deferred) = build_result
    else:
        (stop_binned_spikes, init_stop_binned_feats, offset_log, stop_meta_used,
         stop_meta_groups) = build_result
        deferred = None

    # For decoding we skip interactions. For stop_gam (encoding) we also skip:
    # no retry interactions, and no scaling of rcos_stop_* / *:bin* / binary (handled in scale_binned_feats).
    if not for_decoding and not use_encoding_design:
        init_stop_binned_feats = stop_design_for_decoding.add_interaction_columns(
            init_stop_binned_feats
        )


    # For stop_gam: add temporal (rcos_stop_*) and spatial tuning (*:bin*) after scaling.
    if use_encoding_design and deferred is not None:
        stop_binned_feats = pd.concat(
            [init_stop_binned_feats, deferred['temporal_df']], axis=1
        )
        if deferred['tuning_df'] is not None:
            if len(deferred['tuning_df']) != len(stop_binned_feats):
                raise ValueError('Tuning df length mismatch')
            deferred['tuning_df'].index = stop_binned_feats.index
            stop_binned_feats = pd.concat(
                [stop_binned_feats, deferred['tuning_df']], axis=1
            )
        stop_meta_groups = stop_design_for_encoding.rebuild_encoding_groups_after_blocks(
            stop_binned_feats, deferred
        )
        binrange_dict = deferred.get('binrange_dict')
    else:
        stop_binned_feats = stop_design_for_decoding.scale_binned_feats(
            init_stop_binned_feats,
        )
            
    if use_encoding_design:
        return pn, stop_binned_spikes, stop_binned_feats, offset_log, stop_meta_used, stop_meta_groups, init_stop_binned_feats, binrange_dict
    else:
        return pn, stop_binned_spikes, stop_binned_feats, offset_log, stop_meta_used, stop_meta_groups, None, None
