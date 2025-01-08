from decision_making_analysis.GUAT import GUAT_and_TAFT
from data_wrangling import specific_utils

import os
from os.path import exists
import pandas as pd
import gc



def make_sessions_df_for_one_monkey(raw_data_dir_name, monkey_name):
    sessions_df = specific_utils.initialize_monkey_sessions_df(raw_data_dir_name=raw_data_dir_name)
    sessions_df_for_one_monkey = sessions_df[sessions_df['monkey_name']==monkey_name].copy()
    sessions_df_for_one_monkey.sort_values(by='data_name', inplace=True)
    return sessions_df_for_one_monkey


def collect_info_from_all_sessions(sessions_df_for_one_monkey, 
                                   data_category='decision_making',
                                   data_folder_name='GUAT_info',
                                    df_names=['GUAT_alt_ff_info', 'GUAT_current_ff_info', 'traj_data_df', 'more_traj_data_df', 'more_ff_df', 'traj_ff_info'],
                                    point_index_column_name='point_index'):
    
    all_important_info = dict()
    all_point_index_to_new_number = pd.DataFrame()
    new_point_index_start = 0

    sessions_df_for_one_monkey['retrieved'] = False

    for index, row in sessions_df_for_one_monkey.iterrows():
        if row['retrieved'] == True:
            print('Session {} has been finished. Skip.'.format(row['data_name']))
            continue 

        folder = os.path.join(f'all_monkey_data/{data_category}', row['monkey_name'], row['data_name'], data_folder_name)
        important_info = dict()
        for df in df_names:
            important_info[df] = pd.read_csv(os.path.join(folder, df+'.csv'))

        print('folder:', folder)
        important_info, point_index_to_new_number_df = GUAT_and_TAFT.update_point_index_of_important_df_in_important_info_func(important_info, new_point_index_start, point_index_column_name=point_index_column_name)
        new_point_index_start = point_index_to_new_number_df['new_number'].max()+1
        all_important_info[row['data_name']] = important_info

        point_index_to_new_number_df['data_name'] = row['data_name']
        if len(all_point_index_to_new_number) == 0:
            all_point_index_to_new_number = point_index_to_new_number_df.copy()
        else:
            all_point_index_to_new_number = pd.concat([all_point_index_to_new_number, point_index_to_new_number_df], axis=0)

        gc.collect()

    return all_important_info, all_point_index_to_new_number                                


def check_which_df_exists_for_each_session(sessions_df_for_one_monkey, 
                                           data_category='decision_making',
                                           data_folder_name='GUAT_info',
                                    df_names=['GUAT_alt_ff_info', 'GUAT_current_ff_info', 'traj_data_df', 'more_traj_data_df', 'more_ff_df', 'traj_ff_info']):
    # first assess which df need to be remade
    for name in df_names:
        sessions_df_for_one_monkey[name] = False

    for index, row in sessions_df_for_one_monkey.iterrows():
        folder = os.path.join(os.path.join(f'all_monkey_data/{data_category}', row['monkey_name'], row['data_name'], data_folder_name))
        os.makedirs(folder, exist_ok=True)
        # get the list of df names in the folder
        df_in_folder = os.listdir(folder)
        for df in df_names:
            if df in df_in_folder:
                sessions_df_for_one_monkey.loc[index, df] = True
    return sessions_df_for_one_monkey


def turn_all_important_info_into_combined_info(all_important_info, folder_name, save_each_df_as_csv=False):
    combined_info = dict()
    for data_name, info_dict in all_important_info.items():
        for df_name, df in info_dict.items():
            if df_name in combined_info.keys():
                combined_info[df_name] = pd.concat([combined_info[df_name], df], axis=0)
            else:
                combined_info[df_name] = df.copy()

    for df_name, df in combined_info.items():
        df.reset_index(drop=True, inplace=True)
        if save_each_df_as_csv:
            filepath = os.path.join(folder_name, df_name+'.csv')
            # make sure filepath exists. If not, creates it
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath)

    return combined_info


def try_to_retrieve_combined_info(folder_name, df_names=['GUAT_alt_ff_info', 'GUAT_current_ff_info', 'traj_data_df', 'more_traj_data_df', 'more_ff_df', 'traj_ff_info']):
    combined_info = dict()
    # retrieve all the csv in raw_data_dir_name and put into combined_info
    collect_info_flag = False
    for df_name in df_names:
        filename = df_name + '.csv'
        if not exists(os.path.join(folder_name, filename)):
            # raise a warning
            print("Warning: {} does not exist".format(filename))
            collect_info_flag = True
            break
        else:
            filepath = os.path.join(folder_name, filename)
            print('Retrieved:', df_name)
            combined_info[df_name] = pd.read_csv(filepath)
    return combined_info, collect_info_flag













