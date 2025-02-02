import sys
from planning_analysis.show_planning.get_stops_near_ff import stops_near_ff_based_on_ref_class, find_stops_near_ff_utils
from planning_analysis.show_planning import show_planning_class, nxt_ff_utils
from planning_analysis.only_cur_ff import only_cur_ff_utils
from planning_analysis.plan_factors import test_vs_control_utils
from planning_analysis.agent_analysis import agent_plan_factors_x_sess_class
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, plot_variations_utils, process_variations_utils
from machine_learning.RL.SB3 import rl_for_multiff_utils
from data_wrangling import specific_utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from math import pi
import itertools
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression


def examine_and_compare_columns_between_two_df(df1, df2, verbose=True,
                                               columns_to_exclude_from_comparison=[
                                                   'monkey_name'],
                                               ):

    columns_differed = []
    shared_varients = {}

    varied_columns_1 = process_variations_utils.examine_columns(
        df1, verbose=False)
    varied_columns_2 = process_variations_utils.examine_columns(
        df2, verbose=False)
    all_varied_columns = list(set(varied_columns_1 + varied_columns_2))
    all_varied_columns = [
        col for col in all_varied_columns if col not in columns_to_exclude_from_comparison]

    if verbose:
        print('Columns that have different variations between the two DataFrames:')
    for column in all_varied_columns:
        df1_variants = set(df1[column].unique())
        df2_variants = set(df2[column].unique())
        if df1_variants != df2_variants:
            if verbose:
                print(' ')
                print('===================================================')
                print(' ')
                print('Column:', column)
                print('df1:', df1_variants)
                print('df2:', df2_variants)
            columns_differed.append(column)
            shared_varients[column] = df1_variants.intersection(df2_variants)

    return columns_differed, shared_varients


def _match_combo_of_columns(df1, df2, combo_of_columns_to_match):
    combo_of_columns_to_match = [col for col in combo_of_columns_to_match if (
        col in df1.columns) and (col in df2.columns)]
    new_df1 = pd.DataFrame()
    new_df2 = pd.DataFrame()
    for index, combo in df1[combo_of_columns_to_match].drop_duplicates().iterrows():
        df2_sub = df2[df2[combo_of_columns_to_match].eq(combo).all(axis=1)]
        if df2_sub.shape[0] == 0:
            continue
        new_df1 = pd.concat(
            [new_df1, df1[df1[combo_of_columns_to_match].eq(combo).all(axis=1)]], axis=0)
        new_df2 = pd.concat([new_df2, df2_sub], axis=0)

    new_df1 = new_df1.reset_index(drop=True)
    new_df2 = new_df2.reset_index(drop=True)
    return new_df2


def make_both_players_df(monkey_df, agent_df,
                         combo_of_columns_to_match=['cluster_to_keep', 'cluster_for_interaction', 'use_combd_features_for_cluster_only', 'ref_columns_only']):
    columns_differed, shared_varients = examine_and_compare_columns_between_two_df(
        monkey_df, agent_df, verbose=False)

    monkey_df_sub = monkey_df.copy()
    agent_df_sub = agent_df.copy()
    for key, value in shared_varients.items():
        monkey_df_sub = monkey_df_sub[monkey_df_sub[key].isin(value)]
        agent_df_sub = agent_df_sub[agent_df_sub[key].isin(value)]

    agent_df_sub = _match_combo_of_columns(
        monkey_df_sub, agent_df_sub, combo_of_columns_to_match)

    monkey_df_sub['monkey_or_agent'] = 'monkey'
    agent_df_sub['monkey_or_agent'] = 'agent'
    both_players_df = pd.concat(
        [monkey_df_sub, agent_df_sub], axis=0).reset_index(drop=True)

    # check to see if there are any columns that are not in both players
    columns_not_in_both_players = [
        col for col in monkey_df.columns if col not in both_players_df.columns]
    if len(columns_not_in_both_players) > 0:
        print('Columns that are not in both players:',
              columns_not_in_both_players)

    return both_players_df
