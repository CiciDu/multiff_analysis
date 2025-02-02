
def remove_features_from_extra_clusters(x_features_df):
    # remove_set = []
    # for n_cm in [75, 150, 250]:
    #     column = f'start_dist{n_cm}'
    #     remove_set.extend([col for col in x_features_df.columns if (column in col)])
    # for n_angle in [5, 10, 15]:
    #     column = f'start_ang{n_angle}'
    #     remove_set.extend([col for col in x_features_df.columns if (column in col)])

    # remove_set = [col for col in x_features_df.columns if ('start' in col) or ('flash_cluster' in col) or ('cur_ff_cluster_200' in col)]
    # remove_set = [col for col in x_features_df.columns if ('start_dist' in col) or ('flash_cluster' in col)]
    remove_set = [col for col in x_features_df.columns if (
        'start_dist' in col)]

    return remove_set


def get_minimal_features_to_keep(x_features_df, for_classification=False, deal_with_combd_min_ff_distance=True):

    remove_set = []
    for feature in ['LEFTMOST_ff_angle',
                    'RIGHTMOST_ff_angle',
                    'EARLIEST_FLASH_earliest_flash_rel_time',
                    'LATEST_FLASH_latest_flash_rel_time',
                    'LONGEST_FLASH_flash_duration',
                    'EARLIEST_VIS_earliest_vis_rel_time',
                    'LATEST_VIS_latest_vis_rel_time',
                    'LONGEST_VIS_vis_duration']:
        remove_set.extend(
            [column for column in x_features_df.columns if (feature in column)])

    # remove_set = [column for column in x_features_df.columns if ('combd_' in column) &
    #                                                             ('combd_median_ff_angle' not in column) &
    #                                                             ('combd_median_angle_diff_boundary' not in column) &
    #                                                             ('combd_median_ff_distance' not in column) &
    #                                                             ('combd_total_flash_duration' not in column)
    #                                                             ]
    # # remove_set.extend([column for column in x_features_df.columns if ('cur_ff_cluster_50' in column) & ('ff_distance' in column) & ('combd_median_ff_distance' not in column)])
    # # remove_set.extend([column for column in x_features_df.columns if ('cur_ff_cluster_50' in column) & ('angle_diff_boundary' in column)
    # #                                                                 & ('LONGEST_FLASH_angle_diff_boundary' not in column) &
    # #                                                                 ('combd_median_angle_diff_boundary' not in column)])

    # if both ff_angle_boundary and angle_diff_boundary are present, remove the former
    if any('ff_angle_boundary' in col for col in x_features_df.columns) & any('angle_diff_boundary' in col for col in x_features_df.columns):
        remove_set.extend(
            [column for column in x_features_df.columns if ('ff_angle_boundary' in column)])

    more_remove_set = remove_features_from_extra_clusters(x_features_df)
    remove_set.extend(more_remove_set)

    minimal_features_to_keep = [
        item for item in x_features_df.columns if item not in remove_set]

    if for_classification:
        if ('dir_from_cur_ff_to_nxt_ff' in x_features_df.columns) & ('dir_from_cur_ff_to_nxt_ff' not in minimal_features_to_keep):
            minimal_features_to_keep.append('dir_from_cur_ff_to_nxt_ff')

    if deal_with_combd_min_ff_distance:
        minimal_features_to_keep = deal_with_combd_min_ff_distance_func(
            minimal_features_to_keep)
    return minimal_features_to_keep


def deal_with_combd_min_ff_distance_func(minimal_features_to_keep):
    for ff in ['cur_ff_cluster',
               'cur_ff_ang_cluster',
               'nxt_ff']:
        columns = [col for col in minimal_features_to_keep if (
            ff in col) & ('combd_min_ff_distance' in col)]
        if len(columns) > 1:
            # keep the column with the smallest radius
            columns = sorted(columns, key=lambda x: int(x.split('_')[3]))
            columns_not_to_keep = columns[1:]
            minimal_features_to_keep = [
                item for item in minimal_features_to_keep if item not in columns_not_to_keep]
    return minimal_features_to_keep


def get_reasonable_features_to_keep(x_features_df):

    remove_set = [column for column in x_features_df.columns if ('combd_' in column) &
                                                                ('combd_median_ff_angle' not in column) &
                                                                ('combd_median_angle_diff_boundary' not in column) &
                                                                ('combd_median_ff_distance' not in column) &
                                                                ('combd_total_flash_duration' not in column)
                  ]
    remove_set.extend([column for column in x_features_df.columns if ('cur_ff_cluster_50' in column) & (
        'ff_distance' in column) & ('combd_median_ff_distance' not in column)])
    remove_set.extend([column for column in x_features_df.columns if ('cur_ff_cluster_50' in column) & ('angle_diff_boundary' in column)
                       & ('LONGEST_FLASH_angle_diff_boundary' not in column) &
                       ('combd_median_angle_diff_boundary' not in column)])

    more_remove_set = remove_features_from_extra_clusters(x_features_df)
    remove_set.extend(more_remove_set)

    reasonable_features_to_keep = [
        item for item in x_features_df.columns if item not in remove_set]
    return reasonable_features_to_keep


def get_features_to_keep_based_on_specific_selections():
    features_to_keep = dict()

    radius = 50
    features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_ff_distance',
                                f'cur_ff_cluster_{radius}_LONGEST_FLASH_angle_diff_boundary',
                                ]

    for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
        for feature in ['ff_angle', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
            features_to_keep[radius].append(
                f'cur_ff_cluster_{radius}_{ff}_{feature}')

    radius = 100
    features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_total_flash_duration',
                                f'cur_ff_cluster_{radius}_LONGEST_FLASH_ff_distance'
                                ]

    for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
        for feature in ['ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
            features_to_keep[radius].append(
                f'cur_ff_cluster_{radius}_{ff}_{feature}')

    remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_ff_angle',
                  f'cur_ff_cluster_{radius}_LATEST_FLASH_ff_angle',
                  ]
    features_to_keep[radius] = [
        item for item in features_to_keep[radius] if item not in remove_set]

    radius = 150
    features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_angle_diff_boundary',
                                f'cur_ff_cluster_{radius}_combd_total_flash_duration',
                                ]

    for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
        for feature in ['ff_distance',  'ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
            features_to_keep[radius].append(
                f'cur_ff_cluster_{radius}_{ff}_{feature}')
    remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_ff_angle',
                  f'cur_ff_cluster_{radius}_LATEST_FLASH_ff_angle',
                  ]
    features_to_keep[radius] = [
        item for item in features_to_keep[radius] if item not in remove_set]

    radius = 200
    features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_angle_diff_boundary',
                                f'cur_ff_cluster_{radius}_combd_total_flash_duration',
                                ]

    for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
        for feature in ['ff_distance',  'ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
            features_to_keep[radius].append(
                f'cur_ff_cluster_{radius}_{ff}_{feature}')

    remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_ff_angle',
                  f'cur_ff_cluster_{radius}_LATEST_FLASH_ff_angle',
                  ]
    features_to_keep[radius] = [
        item for item in features_to_keep[radius] if item not in remove_set]

    radius = 250
    features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_angle_diff_boundary',
                                f'cur_ff_cluster_{radius}_combd_total_flash_duration',
                                ]

    for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
        for feature in ['ff_distance',  'ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
            features_to_keep[radius].append(
                f'cur_ff_cluster_{radius}_{ff}_{feature}')

    remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_ff_angle',
                  f'cur_ff_cluster_{radius}_LATEST_FLASH_ff_angle',
                  ]
    features_to_keep[radius] = [
        item for item in features_to_keep[radius] if item not in remove_set]

    radius = 300
    features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_angle_diff_boundary',
                                f'cur_ff_cluster_{radius}_combd_total_flash_duration',
                                ]

    for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
        for feature in ['ff_distance',  'ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
            features_to_keep[radius].append(
                f'cur_ff_cluster_{radius}_{ff}_{feature}')

    remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_ff_angle',
                  f'cur_ff_cluster_{radius}_LATEST_FLASH_ff_angle',
                  ]
    features_to_keep[radius] = [
        item for item in features_to_keep[radius] if item not in remove_set]
    return features_to_keep


# def get_features_to_keep_based_on_specific_selections():
#     features_to_keep = dict()

#     radius = 50
#     features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_ff_distance',
#                                 f'cur_ff_cluster_{radius}_LONGEST_FLASH_angle_diff_boundary',
#                                 ]

#     for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
#         for feature in ['ff_angle', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
#             features_to_keep[radius].append(f'cur_ff_cluster_{radius}_{ff}_{feature}')


#     radius = 100
#     features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_total_flash_duration',
#                                 f'cur_ff_cluster_{radius}_LONGEST_FLASH_ff_distance'
#                                 ]

#     for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
#         for feature in ['ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
#             features_to_keep[radius].append(f'cur_ff_cluster_{radius}_{ff}_{feature}')

#     remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_angle_diff_boundary',
#                     f'cur_ff_cluster_{radius}_LATEST_FLASH_angle_diff_boundary',
#                     ]
#     features_to_keep[radius] = [item for item in features_to_keep[radius] if item not in remove_set]


#     radius = 150
#     features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_angle_diff_boundary',
#                                             f'cur_ff_cluster_{radius}_combd_total_flash_duration',
#                                             ]

#     for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
#         for feature in ['ff_distance',  'ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
#             features_to_keep[radius].append(f'cur_ff_cluster_{radius}_{ff}_{feature}')
#     remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_angle_diff_boundary',
#                     f'cur_ff_cluster_{radius}_LATEST_FLASH_angle_diff_boundary',
#                     ]
#     features_to_keep[radius] = [item for item in features_to_keep[radius] if item not in remove_set]


#     radius = 200
#     features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_angle_diff_boundary',
#                                             f'cur_ff_cluster_{radius}_combd_total_flash_duration',
#                                             ]

#     for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
#         for feature in ['ff_distance',  'ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
#             features_to_keep[radius].append(f'cur_ff_cluster_{radius}_{ff}_{feature}')

#     remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_angle_diff_boundary',
#                     f'cur_ff_cluster_{radius}_LATEST_FLASH_angle_diff_boundary',
#                     ]
#     features_to_keep[radius] = [item for item in features_to_keep[radius] if item not in remove_set]


#     radius = 250
#     features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_angle_diff_boundary',
#                                             f'cur_ff_cluster_{radius}_combd_total_flash_duration',
#                                             ]

#     for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
#         for feature in ['ff_distance',  'ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
#             features_to_keep[radius].append(f'cur_ff_cluster_{radius}_{ff}_{feature}')

#     remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_angle_diff_boundary',
#                     f'cur_ff_cluster_{radius}_LATEST_FLASH_angle_diff_boundary',
#                     ]
#     features_to_keep[radius] = [item for item in features_to_keep[radius] if item not in remove_set]


#     radius = 300
#     features_to_keep[radius] = [f'cur_ff_cluster_{radius}_combd_median_angle_diff_boundary',
#                                             f'cur_ff_cluster_{radius}_combd_total_flash_duration',
#                                             ]

#     for ff in ['LEFTMOST', 'RIGHTMOST', 'EARLIEST_FLASH', 'LATEST_FLASH', 'LONGEST_FLASH']:
#         for feature in ['ff_distance',  'ff_angle', 'angle_diff_boundary', 'flash_duration', 'earliest_flash_rel_time', 'latest_flash_rel_time']:
#             features_to_keep[radius].append(f'cur_ff_cluster_{radius}_{ff}_{feature}')

#     remove_set = [f'cur_ff_cluster_{radius}_LONGEST_FLASH_angle_diff_boundary',
#                     f'cur_ff_cluster_{radius}_LATEST_FLASH_angle_diff_boundary',
#                     ]
#     features_to_keep[radius] = [item for item in features_to_keep[radius] if item not in remove_set]
#     return features_to_keep
