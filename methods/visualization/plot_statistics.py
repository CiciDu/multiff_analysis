import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')
import os
import seaborn as sns
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.express as px
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MaxNLocator
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from matplotlib.ticker import FixedLocator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.ticker import FixedLocator




def plot_distribution(var, xlim=None, x_of_vline=None, scale_factor=1, plot_cdf=True):
    plt.figure(figsize=(10, 7))
    if xlim is not None:
        plt.xlim(xlim)
    sns.histplot(var*scale_factor)
    plt.show()
    

    if plot_cdf:
        plt.figure(figsize=(10, 7))
        if xlim is not None:
            plt.xlim(xlim)
        if x_of_vline is not None:
            plt.axvline(x=x_of_vline, color='k', linestyle='--', linewidth=1)
        sns.ecdfplot(var*scale_factor)
        plt.show()
        

    
def plot_feature_histograms_for_monkey_and_agent(all_trial_features_valid_monkey, all_trial_features_valid_agent, num_rows=3, num_cols=3, data_folder_name=None, file_name=None):
    """
    For each attribute in all_trial_features_valid (which comes from all_trial_features), plot a histogram that compares the monkey and the agent


    Parameters
    ----------
    all_trial_features_valid_agent: dataframe
        belonging to the agent, containing various characteristics of each trial
    all_trial_features_valid_monkey: dataframe
        belonging to the monkey, containing various characteristics of each trial
    data_folder_name: str
      name or path of the folder to store the graph

    """

    fig = plt.figure(figsize=(num_rows*3, num_cols*3))
    sns.set_style(style="darkgrid")
    counter = 0


    variable_of_interest = "t_last_visible"
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    sns.histplot(data = all_trial_features_valid_agent[variable_of_interest], kde = False, alpha = 0.4, color = "green", binwidth = 0.1, stat="probability")
    sns.histplot(data = all_trial_features_valid_monkey[variable_of_interest], kde = False, alpha = 0.4, color = "blue", binwidth = 0.1, stat="probability")
    axes.set_title("Time Since Target Last Visible", fontsize=11)
    axes.set_xlim([0, 6])
    axes.set_xlabel("Time (s)", fontsize = 11)
    plt.legend(labels=["Agent(LSTM)", "Monkey"], fontsize=11, loc="upper right")

    variable_of_interest = "num_stops_near_target"
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    sns.histplot(data = all_trial_features_valid_agent[variable_of_interest], kde = False, alpha = 0.3, color = "green", binwidth=0.5, binrange=(-0.25,5.25), stat="probability", edgecolor='grey')
    sns.histplot(data = all_trial_features_valid_monkey[variable_of_interest], kde = False, alpha = 0.3, color = "blue", binwidth=0.5, binrange=(-0.1,5.4), stat="probability", edgecolor='grey')
    axes.set_title("Number of Stops Near Targets", fontsize=11)
    axes.set_xlim(-0.25,5)
    axes.set_xlabel("Number of Stops", fontsize=11)


    variable_of_interest = "n_ff_in_a_row"
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    sns.histplot(data = all_trial_features_valid_agent[variable_of_interest], kde = False, alpha = 0.3, binrange=(-0.25,5.25), color = "green", binwidth=0.5, stat="probability",  edgecolor='grey')
    sns.histplot(data = all_trial_features_valid_monkey[variable_of_interest], kde = False, alpha = 0.3, binrange=(-0.1,5.4), color = "blue", binwidth=0.5, stat="probability",  edgecolor='grey')
    axes.set_xlim(0.25,5.25)
    axes.set_title("Num of FF Caught in a Cluster", fontsize=11)
    axes.set_xlabel("Number of Fireflies", fontsize=11)


    variable_of_interest = "abs_angle_last_visible"
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    sns.histplot(data = all_trial_features_valid_agent[variable_of_interest], kde = False, binwidth=0.02, alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
    sns.histplot(data = all_trial_features_valid_monkey[variable_of_interest], kde = False,  binwidth=0.02, alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
    axes.set_title("Abs Angle of Target Last Visible", fontsize=11)
    axes.set_xlabel("Angle (rad)", fontsize=11)
    axes.set_xticks(np.arange(0.0, 0.9, 0.2))
    axes.set_xticks(axes.get_xticks())
    axes.set_xticklabels(np.arange(0.0, 0.9, 0.2).round(1))
    axes.set_xlim(0, 0.7)


    variable_of_interest = "t"
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    sns.histplot(data = all_trial_features_valid_agent[variable_of_interest], kde = False, binwidth=1,  alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
    sns.histplot(data = all_trial_features_valid_monkey[variable_of_interest], kde = False, binwidth=1,  alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
    axes.set_title("Trial Duration", fontsize=11)
    axes.set_xlabel("Duration (s)", fontsize=11)

    variable_of_interest = "num_stops"
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    sns.histplot(data = all_trial_features_valid_agent[variable_of_interest], binwidth=1, binrange=(0.5, 10.5), alpha = 0.3, color = "green", stat="probability", edgecolor='grey')
    sns.histplot(data = all_trial_features_valid_monkey[variable_of_interest], binwidth=1, binrange=(0.6, 10.6), alpha = 0.3, color = "blue", stat="probability", edgecolor='grey')
    axes.set_xlabel("Number of Stops", fontsize=11)
    axes.set_xlim(0.7, 12)
    axes.set_title("Number of Stops During Trials", fontsize=11)


    variable_of_interest = "d_last_visible"
    counter += 1
    axes = fig.add_subplot(num_rows, num_cols, counter)
    sns.histplot(data = all_trial_features_valid_agent[variable_of_interest]/100, kde = False, alpha = 0.3,  color = "green", binwidth=2, stat="probability",  edgecolor='grey')
    sns.histplot(data = all_trial_features_valid_monkey[variable_of_interest]/100, kde = False, alpha = 0.3,  color = "blue", binwidth=2, stat="probability",  edgecolor='grey')
    axes.set_xlim(0, 50)
    axes.set_title("Distance of Target Last Visible", fontsize=11)
    axes.set_xlabel("Distance (100 cm)", fontsize=11)
    xticklabels=axes.get_xticks().tolist()
    xticklabels = [str(int(label)) for label in xticklabels]
    xticklabels[-1]='50+'
    axes.set_xticks(axes.get_xticks())
    axes.set_xticklabels(xticklabels)


    plt.tight_layout()
    if data_folder_name is not None:
        if not os.path.isdir(data_folder_name):
            os.makedirs(data_folder_name)
        if file_name is None:
            file_name = 'feature_histograms.png'  
        figure_name = os.path.join(data_folder_name, f"{file_name}.png")
        plt.savefig(figure_name)
    plt.show()
    


def plot_merged_df(merged_df, x="Label", y="Rate", hue=None, label_order=None, ax=None):
    sns.set_style(style="darkgrid")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    # grouped barplot
    if hue is not None:
        ax = sns.barplot(x=x, y=y, data=merged_df, hue=hue, order=label_order, ax=ax)
    else:
        ax = sns.barplot(x=x, y=y, data=merged_df, order=label_order, ax=ax)

    xticks = ax.get_xticks()
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=40, ha="right")

    plt.tight_layout()
    return ax


def plot_merged_df_by_category(merged_df, category_column_name, category_order=None, x="Player", y="Rate", hue=None, percentage=False, subplots=True, num_columns=3):
    """
    Make one barplot for each category in the merged_df (can be merged_stats_df or merged_medians_df, and can compare the monkey and the agent(s)

    Parameters
    ----------
    merged_df: dataframe
        containing various characteristics of each trial for both the monkey and the agent(s)
    percentage: bool
        whether to use percentage for the y variable

    """

    if category_order is None:
        category_order = merged_df[category_column_name].unique()
    
    if subplots:
        num_rows = math.ceil(len(category_order)/3)
        fig = plt.figure(figsize=(num_columns*3, num_rows*3.5))
    else:
        plt.figure(figsize=(4, 8))

    for i in range(len(category_order)):
        category = category_order[i]
        category_df = merged_df[merged_df[category_column_name] == category]
        if subplots:
            ax = fig.add_subplot(num_rows, num_columns, i+1)
            ax.set_title(category, fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)      
        else:
            plt.title(category, fontsize=22)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=15) 
        if hue is not None:
            ax = sns.barplot(x=x, y=y, hue=hue, data=category_df)
        else:
            ax = sns.barplot(x=x, y=y, data=category_df)
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        if percentage:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    sns.set_style(style="darkgrid")    
    plt.tight_layout()  


def plot_pattern_frequencies(pattern_frequencies, compare_monkey_and_agent=False, data_folder_name=None, file_name=None, monkey_name='',
                             ax=None, return_ax=False, hue=None):
    subset = pattern_frequencies[pattern_frequencies['Item'].isin(['two_in_a_row', 'visible_before_last_one', 'disappear_latest', 'waste_cluster_around_target',
                                                                   'ignore_sudden_flash', 'give_up_after_trying', 'try_a_few_times', 'ff_capture_rate', 'stop_success_rate'])]
    label_order = ['Visible before last capture', 'Target disappears latest', 'Two in a row', 'Waste cluster around last target', 'Try a few times', 'Give up after trying', 'Ignore sudden flash', 'Firefly capture rate (per s)', 'Stop success rate']
    if compare_monkey_and_agent:
        hue = "Player"
    ax = plot_merged_df(subset, x="Label", y="Rate", hue=hue, label_order = label_order, ax=ax)


    if data_folder_name is not None:
        if not os.path.isdir(data_folder_name):
            os.makedirs(data_folder_name)
        if compare_monkey_and_agent:
            if file_name is None:
                file_name = 'compare_pattern_frequencies.png'
            figure_name = os.path.join(data_folder_name, file_name)
            plt.savefig(figure_name)
        else:
            if file_name is None:
                file_name = 'pattern_frequencies.png'
            figure_name = os.path.join(data_folder_name, file_name)
            plt.savefig(figure_name)

    if monkey_name != '':
        monkey_name = monkey_name + ': '
    ax.set_title(monkey_name + 'Pattern Frequencies', fontsize=15)

    if return_ax:
        return ax
    else:
        plt.show()
    

def plot_feature_statistics(feature_statistics, compare_monkey_and_agent=False, data_folder_name=None, file_name=None, monkey_name='', hue=None):

    label_order = ['time', 'time target last seen', 'abs angle target last seen', 'num stops', 'num stops near target']
    label_order_for_medians = ['Median ' + label for label in label_order]
    label_order_for_means = ['Mean ' + label for label in label_order]

    if monkey_name != '':
        monkey_name = monkey_name + ': '

    if compare_monkey_and_agent:
        plot_merged_df_by_category(feature_statistics, category_column_name = "Label for median", category_order = label_order_for_medians, 
                                   x="Player", y="Median", percentage=False, subplots=True)
        plot_merged_df_by_category(feature_statistics, category_column_name = "Label for mean", category_order = label_order_for_means, 
                                   x="Player", y="Mean", percentage=False, subplots=True)
    
    else:        
        subset = feature_statistics[feature_statistics['Item'].isin(['t', 't_last_visible', 'abs_angle_last_visible', 'num_stops', 'num_stops_near_target'])]

        # Create a 1x2 subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot the first subplot
        plot_merged_df(subset, x="Label for median", y="Median", label_order=label_order_for_medians, ax=ax1, hue=hue)
        ax1.set_title(monkey_name + 'Medians', fontsize=15)
        ax1.set_xlabel('')

        # Plot the second subplot
        plot_merged_df(subset, x="Label for mean", y="Mean", label_order=label_order_for_means, ax=ax2, hue=hue)
        ax2.set_title(monkey_name + 'Means', fontsize=15)
        ax2.set_xlabel('')

        # Adjust layout
        plt.tight_layout()
        plt.show()


    if data_folder_name is not None:
        if not os.path.isdir(data_folder_name):
            os.makedirs(data_folder_name)
        if compare_monkey_and_agent:
            if file_name is None:
                file_name = 'compare_feature_statistics.png'  
            figure_name = os.path.join(data_folder_name, file_name)
            plt.savefig(figure_name)
        else:
            if file_name is None:
                file_name = 'feature_statistics.png'  
            figure_name = os.path.join(data_folder_name, file_name)
            plt.savefig(figure_name)

    plt.show()


def plot_num_ff_caught_in_a_row_in_barplot(pattern_frequencies, show_one_in_a_row = True):
    plt.rcParams['figure.figsize'] = (8, 6)
    sns.set_style(style="darkgrid")
    if 'Item' in pattern_frequencies.columns:
        pattern_frequencies = pattern_frequencies.set_index('Item')
    if show_one_in_a_row:
        temp_df = pattern_frequencies.loc[['one_in_a_row', 'two_in_a_row', 'three_in_a_row', 'four_in_a_row']]
    else:
        temp_df = pattern_frequencies.loc[['two_in_a_row', 'three_in_a_row', 'four_in_a_row']]
    axes = sns.barplot(data = temp_df, x="Label", y='Percentage')
    axes.set(title='Number of Fireflies Captured In a Row', xlabel = 'Number of Fireflies', ylabel='Percentage(%)')
    plt.rcParams.update({'font.size': 15})

    xticks = axes.get_xticks()
    axes.xaxis.set_major_locator(FixedLocator(xticks))
    xticklabels = axes.get_xticklabels()
    axes.set_xticklabels(xticklabels, rotation=20, size=13)

    for p in axes.patches:
        axes.annotate(format(p.get_height(), '.1f') + "%", 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', 
                      xytext = (0, 5), 
                      textcoords = 'offset points', size=11)
        
    plt.show()


def plot_num_ff_caught_in_a_row_in_pie_chart(pattern_frequencies):
    if 'Item' in pattern_frequencies.columns:
        pattern_frequencies = pattern_frequencies.set_index('Item')
    temp_df = pattern_frequencies.loc[['one_in_a_row', 'two_in_a_row', 'three_in_a_row', 'four_in_a_row']]
    temp_df = temp_df[temp_df['Percentage']>0]
    data = list(temp_df['Frequency'])
    colors = sns.color_palette('pastel')[0:5]
    labels = temp_df['Label']
    #create pie chart
    plt.rcParams.update({'font.size': 11})
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%', pctdistance=0.7)
    plt.tight_layout()
    plt.show()


def plot_categorical_variable_in_barplot(all_trial_features, var_of_interest):
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams.update({'font.size': 13})
    temp_df = all_trial_features[[var_of_interest]].copy()
    temp_df = temp_df[temp_df[var_of_interest]!=9999]
    temp_df[var_of_interest] = temp_df[var_of_interest].astype('str')
    temp_df.sort_values(var_of_interest, inplace=True)
    axes = sns.histplot(data=temp_df, x=var_of_interest, stat='percent')
    
    # for p in axes.patches:
    #     axes.annotate(format(p.get_height(), '.1f') + "%", 
    #                   (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                   ha = 'center', va = 'center', 
    #                   xytext = (0, 5), 
    #                   textcoords = 'offset points', size=11)


def plot_num_ff_around_target_in_barplot(all_trial_features):
    plot_categorical_variable_in_barplot(all_trial_features, var_of_interest = 'num_ff_around_target')
    plt.title('Maximum Number of Alive Fireflies Around the Target During a Trial')
    plt.xlabel('Number of Fireflies')
    plt.ylabel('Percentage(%)')    
    plt.show()
    

def plot_num_stops_near_target_in_barplot(all_trial_features):
    #all_trial_features = all_trial_features.sort_values('num_stops_near_target', inplace=True)
    plot_categorical_variable_in_barplot(all_trial_features, var_of_interest = 'num_stops_near_target')
    plt.title('Number of Stops Near Target')
    plt.xlabel('Number of Stops')
    plt.ylabel('Percentage(%)')    
    plt.show()
    

def plot_num_stops_in_histogram(all_trial_features, var_of_interest):
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams.update({'font.size': 13})
    temp_df = all_trial_features[[var_of_interest]]
    temp_df = temp_df[temp_df[var_of_interest] != 9999]
    sns.histplot(all_trial_features, x=var_of_interest, stat='percent', binwidth=1, edgecolor='white', linewidth=0.8)
    plt.xlim((-0.5, 12.5))
    plt.xlabel('Number of Stops')
    plt.ylabel('Percentage(%)')  


def plot_proportion_of_target_closest(target_closest):
    plot_proportion_of_time_points(target_closest)
    plt.title('Proportion of Points Where the Target is the Closest')
    plt.show()
    

def plot_proportion_of_target_angle_smallest(target_angle_smallest):
    plot_proportion_of_time_points(target_angle_smallest)
    plt.title('Proportion of Points Where the Target Has the Smallest Angle')
    plt.show()
    

def plot_proportion_of_time_points(list_of_values_for_points):
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['figure.figsize'] = (7, 7)
    prop_points = sum([1 if point > 1.9 else 0 for point in list_of_values_for_points])/len(list_of_values_for_points)
    prop_points_com = 1 - prop_points
    labels = ['Yes', 'No']
    data = [prop_points, prop_points_com]
    colors = sns.color_palette('pastel')[0:2]
    #create pie chart
    plt.rcParams.update({'font.size': 11})
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%', pctdistance=0.7)
    plt.tight_layout()


def plot_number_of_visible_ff_per_point_in_histogram(ff_dataframe):
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible']==1]
    num_ff_vis = ff_dataframe_visible[['point_index', 'ff_index']].groupby('point_index').nunique()
    fig = px.histogram(num_ff_vis, x="ff_index")
    fig.update_layout(
        title_text='Number of Visible Fireflies at Any Point', # title of plot
        xaxis_title_text='Number of Visible Fireflies', # xaxis label
        yaxis_title_text='Percentage of Points', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        #bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    fig.show()


def plot_number_of_ff_in_memory_per_point_in_histogram(ff_dataframe):
    ff_dataframe_in_memory = ff_dataframe[ff_dataframe['visible']==0]
    num_ff_in_memory = ff_dataframe_in_memory[['point_index', 'ff_index']].groupby('point_index').nunique()
    fig = px.histogram(num_ff_in_memory, x="ff_index")
    fig.update_layout(
        title_text='Number of Fireflies In Memory at Any Point', # title of plot
        xaxis_title_text='Number of Fireflies In Memory', # xaxis label
        yaxis_title_text='Percentage of Points', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        #bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    fig.show()


 # First find the points where the target and at least 1 non-target is present in ff_dataframe
def compare_target_with_non_targets(ff_dataframe, var_of_interest='ff_distance'):
    sns.set_style(style="darkgrid")
    plt.rcParams['figure.figsize'] = (8, 6)
    target_present_df = ff_dataframe[ff_dataframe['ff_index'] == ff_dataframe['target_index']]
    target_present_points = np.array(target_present_df.point_index)
    non_target_present_df = ff_dataframe[ff_dataframe['ff_index'] != ff_dataframe['target_index']]
    non_target_present_points = np.array(non_target_present_df.point_index)
    both_present_points = np.intersect1d(target_present_points, non_target_present_points)

    target_present_df = target_present_df[target_present_df['point_index'].isin(both_present_points)]
    target_values = np.array(target_present_df.sort_values('point_index')[var_of_interest])
    non_target_present_df = non_target_present_df[non_target_present_df['point_index'].isin(both_present_points)]
    # Find the minimum var_of_interest value for each time point for the non-targets
    non_target_present_df = non_target_present_df[['point_index', var_of_interest]].groupby('point_index').min().sort_values('point_index')
    non_target_min_values = np.array(non_target_present_df.sort_values('point_index')[var_of_interest])
    dif = non_target_min_values - target_values

    if (var_of_interest == 'ff_angle') or (var_of_interest == 'ff_angle_boundary'):
        dif = dif*180/math.pi
    
    sns.histplot(dif, stat="percent")
    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('Number of Trials')
    dif_mean = round(np.mean(np.array(dif)), 2)
    dif_std = round(np.std(np.array(dif)), 2)
    if var_of_interest == 'ff_distance':
        plt.title('Differences in Fireflies\'s Distances: ' + f'$\u03bc ={dif_mean}, \u03C3 ={dif_std} $', y=1.05)
        plt.xlabel('Distance of Non-Target - Distance of Target (°)')
    elif var_of_interest == 'ff_angle':
        plt.title('Differences in Fireflies\'s Angles: ' + f'$\u03bc ={dif_mean}\N{DEGREE SIGN}, \u03C3 ={dif_std}\N{DEGREE SIGN} $', y=1.05)
        plt.xlabel('Absolute Angle of Non-Target - Absolute Angle of Target (°)')
    elif var_of_interest == 'ff_angle_boundary':
        plt.title('Differences in Fireflies\'s Angles to Reward Boundaries: ' + f'$\u03bc ={dif_mean}\N{DEGREE SIGN}, \u03C3 ={dif_std}\N{DEGREE SIGN} $', y=1.05)
        plt.xlabel('Absolute Angle of Non-Target - Absolute Angle of Target (°)')
    plt.show()
    

def fit_and_plot_linear_regression(x_array, y_array, show_regression=True, show_r=True, print_r_squared=True, hue=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    matplotlib.rcParams.update({'font.size': 15})
    model = LinearRegression()
    model.fit(x_array.reshape((-1, 1)), y_array)
    r_squared = model.score(x_array.reshape((-1, 1)), y_array)
    r = math.sqrt(r_squared)
    p1 = sns.scatterplot(x=x_array, y=y_array, hue=hue, ax=ax, s=50)
    if print_r_squared:
        print('coefficient of determination:', r_squared)
    if show_regression:
        p2 = sns.lineplot(x=x_array, y=model.coef_*x_array+model.intercept_, ax=ax)
    if show_r:
        text_x_position=min(x_array) + 0.1*(max(x_array)-min(x_array))
        text_y_position=min(y_array) + 0.8*(max(y_array)-min(y_array))
        plt.text(text_x_position, text_y_position, f'$r = {round(r, 2)}$')


def plot_correlations_in_record(df, parameter_columns = ['action_noise_std', 'ffxy_noise_std', 'num_obs_ff', 'max_in_memory_time'], 
                                outcome_columns = None, color_column=None, show_regression = True, show_r = True, print_r_squared = False):
    sns.set_style(style="darkgrid")
    matplotlib.rcParams.update({'font.size': 15})

    if outcome_columns is None:
        outcome_columns = np.setdiff1d(df.columns, parameter_columns)
    if isinstance(color_column, str):
        color_column = np.array(df[color_column])
    for i in range(len(outcome_columns)):
        outcome_name = outcome_columns[i]
        fig = plt.figure(figsize=(15,5))
        fig.suptitle(outcome_name)
        for j in range(len(parameter_columns)):
            parameter_name = parameter_columns[j]
            x_array = np.array(df[parameter_name])
            y_array = np.array(df[outcome_name])
            axes = fig.add_subplot(1, 4, j+1)
            fit_and_plot_linear_regression(x_array, y_array, show_regression, show_r, print_r_squared, ax=axes, hue=color_column)
            plt.xlabel(parameter_name)
            if (parameter_name == 'max_in_memory_time') or (parameter_name == 'num_obs_ff'):
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            #plt.ylabel(outcome_name)
        plt.tight_layout()
        plt.show()
        
