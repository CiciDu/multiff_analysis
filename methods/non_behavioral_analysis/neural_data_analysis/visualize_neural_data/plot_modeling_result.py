import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import colorcet
import logging
from matplotlib import rc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import rcca



# Function to make a series of bar plots of ranked loadings
def make_a_series_of_barplots_of_ranked_loadings_or_weights(squared_loading, canon_corr, num_variates, 
                                                 keep_one_value_for_each_feature=False, 
                                                 max_plots_to_show=None,
                                                 max_features_to_show_per_plot=20, 
                                                 horizontal_bars=True):
    # Get the unique feature categories
    unique_feature_category = get_unique_feature_category(squared_loading, num_variates, keep_one_value_for_each_feature, max_features_to_show_per_plot)
    # Generate a color dictionary for the unique feature categories
    color_dict = get_color_dict(unique_feature_category)

    if max_plots_to_show is None:
        max_plots_to_show = num_variates
    # Iterate over the number of variates
    for variate in range(max_plots_to_show):
        # If the flag is set to keep one value for each feature
        if keep_one_value_for_each_feature:
            # Sort the squared loadings, group by feature category, and keep the first (max) value
            loading_subset = squared_loading.sort_values(variate, ascending=False, key=abs).groupby('feature_category').first().reset_index(drop=False)
        else:
            # Otherwise, just copy the squared loadings
            loading_subset = squared_loading.copy()
        # Sort the subset by the variate and keep the top features up to the max limit
        loading_subset = loading_subset.sort_values(by=variate, ascending=False, key=abs).iloc[:max_features_to_show_per_plot]

        # Create a new plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # If horizontal bars are preferred
        if horizontal_bars:
            # Create a horizontal bar plot with seaborn
            sns.barplot(data=loading_subset, x=variate, y='feature', dodge=False, ax=ax, hue='feature_category', palette=color_dict, orient='h')
            plt.xlabel("Squared Loading", fontsize=14)
            plt.ylabel("")
            ax.tick_params(axis='x', which='major', labelsize=14)
            ax.tick_params(axis='y', which='major', labelsize=14)
        else:
            # Otherwise, create a vertical bar plot
            sns.barplot(data=loading_subset, x='feature', y=variate, dodge=False, ax=ax, hue='feature_category', palette=color_dict)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Squared Loading", fontsize=14)
            ax.tick_params(axis='x', which='major', labelsize=14)
            ax.tick_params(axis='y', which='major', labelsize=14)

        # If the flag is set to keep one value for each feature, remove the legend
        if keep_one_value_for_each_feature:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Draw a vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='--')
        
        # Calculate the coefficient and set the title of the plot
        coefficient = np.around(np.array(canon_corr), 2)[variate]
        plt.title(f'Variate: {variate + 1}; canonical correlation coefficient: {coefficient}', fontsize=18)
        
        # Display the plot
        plt.show()
        
        # Close the plot to free up memory
        

def plot_pgam_tuning_functions(res, indices_of_vars_to_plot=None):
    if indices_of_vars_to_plot is None:
        indices_of_vars_to_plot = np.arange(len(res['x_kernel']))


    # each row of res contains the info about a variable
    # some info are shared for all the variables (p-rsquared for example is a goodness of fit measure for the model
    # it is shared, not a property of the variable), while other, like the parameters of the b-splines, 
    # are variable specific

    # print('\n\n')
    # print('Result structarray types\n========================\n')
    # for name in res.dtype.names: 
    #     print('%s: \t %s'%(name, type(res[name][0])))


    num_vars = len(indices_of_vars_to_plot)
    num_var_per_plot = 3
    num_plots = math.ceil(num_vars/num_var_per_plot)
    var_counter = 0
    var_index = indices_of_vars_to_plot[var_counter]
    # plot tuning functions

    for j in range(num_plots):
        plt.figure(figsize=(5*num_var_per_plot,7))
        for k in range(num_var_per_plot):
            plt.subplot(2, num_var_per_plot,k+1)
            #plt.title(str(var_index) + ', log-space %s'%res['variable'][var_index])
            plt.title('log-space %s'%res['variable'][var_index])
            x_kernel = res['x_kernel'][var_index]
            y_kernel = res['y_kernel'][var_index]
            ypCI_kernel = res['y_kernel_pCI'][var_index]
            ymCI_kernel = res['y_kernel_mCI'][var_index]
            
            plt.plot(x_kernel.reshape(-1), y_kernel.reshape(-1), color='r')
            plt.fill_between(x_kernel.reshape(-1), ymCI_kernel.reshape(-1), ypCI_kernel.reshape(-1), color='r', alpha=0.3)
            
            
            x_firing = res['x_rate_Hz'][var_index]
            # y_firing_model = res['model_rate_Hz'][var_index]
            # y_firing_raw = res['raw_rate_Hz'][var_index]
            y_firing_model = res['y_rate_Hz_model'][var_index]
            y_firing_raw = res['y_rate_Hz_raw'][var_index]


            plt.subplot(2,num_var_per_plot,k+1+num_var_per_plot)
            plt.title('rate-space %s'%res['variable'][var_index])
            
            # modify the code below so that it makes plot despite NANs
            
            plt.plot(x_firing[0], y_firing_raw.reshape(-1), 'o-', markersize=2, color='k',label='raw')
            plt.plot(x_firing[0], y_firing_model.reshape(-1), 'o-', markersize=2, color='r',label='model')
            
            
            plt.legend()
            plt.tight_layout()

            var_counter += 1
            if var_counter == num_vars:
                break
            var_index = indices_of_vars_to_plot[var_counter]
        plt.show()


# Function to get unique feature categories based on the given parameters
def get_unique_feature_category(squared_loading, num_variates, keep_one_value_for_each_feature, max_features_to_show_per_plot):
    unique_feature_category = np.array([])
    for variate in range(num_variates):
        # If the flag is set to keep one value for each feature
        if keep_one_value_for_each_feature:
            # Sort the squared loadings, group by feature category, and keep the first (max) value
            loading_subset = squared_loading.sort_values(variate, ascending=False, key=abs).groupby('feature_category').first().reset_index(drop=False)
        else:
            # Otherwise, just copy the squared loadings
            loading_subset = squared_loading.copy()
        # Sort the subset by the variate and keep the top features up to the max limit
        loading_subset = loading_subset.sort_values(by=variate, ascending=False, key=abs).iloc[:max_features_to_show_per_plot]
        # Update the unique feature categories
        unique_feature_category = np.unique(np.concatenate([unique_feature_category, loading_subset.feature_category]))
    # Log the number of unique feature categories included in the plot
    logging.info(f"{len(unique_feature_category)} out of {len(squared_loading.feature_category.unique())} feature categories are included in the plot")
    return unique_feature_category

# Function to generate a color dictionary for the unique feature categories
def get_color_dict(unique_feature_category):
    # Get the first 10 colors from the Set3 palette
    qualitative_colors = sns.color_palette("Set3", 10)
    # Get the remaining colors from the Glasbey palette
    qualitative_colors_2 = sns.color_palette(colorcet.glasbey, n_colors=len(unique_feature_category)-10)
    # Combine the two color palettes
    qualitative_colors.extend(qualitative_colors_2)
    # Create a dictionary mapping each feature category to a color
    color_dict = {unique_feature_category[i]: qualitative_colors[i] for i in range(len(unique_feature_category))}
    return color_dict


def plot_smoothed_temporal_feature(df, column, sm_handler, kernel_h_length):
    event = df[column].values
   
    # Retrieve the B-spline convolved with the "event" variable
    convolved_ev = sm_handler[column].X.toarray()

    # Retrieve the B-spline used for the convolution
    basis = sm_handler[column].basis_kernel.toarray()

    # Get the x values for the 1st subplot
    tps = np.repeat(np.arange(kernel_h_length) - kernel_h_length // 2, basis.shape[1]).reshape(basis.shape)

    # Plot the basis & the convolved events
    plt.figure(figsize=(8, 2.5))

    # Plot the basis for the kernel h
    plt.subplot(121)
    plt.title('Kernel Basis')
    plt.plot(tps, basis)
    plt.xlabel('Time Points')

    # Select an interval containing an event
    event_time_points = np.where(event == 1)[0]
    # Select the first event that occurs after 150 time points
    event_time_points = event_time_points[event_time_points > 150]
    if len(event_time_points) == 0:
        raise ValueError('No event found in specified time interval')
    idx0, idx1 = event_time_points[0] - 100, event_time_points[0] + 400

    # Extract the events convolved with each of the B-spline elements
    conv = convolved_ev[idx0:idx1, :]

    # Get the x values for the 2nd subplot
    tps = np.arange(0, idx1 - idx0) # - 100
    tps = np.repeat(tps, conv.shape[1]).reshape(conv.shape)

    # Plot the convolved events
    plt.subplot(122)
    plt.title('Convolved Events')
    plt.plot(tps, conv)
    plt.title(column)
    plt.vlines(tps[0, 0] + np.where(event[idx0:idx1])[0], 0, 1.5, 'k', ls='--', label='Event')
    plt.xlabel('Time Points')
    plt.legend()
    plt.show()
    plt.close()


def plot_smoothed_spatial_feature(df, column, sm_handler):
    # Retrieve the B-spline evaluated at x
    X_1D = sm_handler[column].X.toarray()
    
    # Get a sorted version of the variable
    column_values = df[column].values
    idx_srt = np.argsort(column_values)
    X_srt = X_1D[idx_srt]
    
    # Plot the B-spline basis functions
    plt.figure(figsize=(15, 3))
    plt.subplot(131)
    plt.title(column)
    plt.plot(column_values[idx_srt], X_srt)
    
    # Unordered scatter plot
    plt.subplot(132)
    plt.title('Unordered scatter plot')
    plt.scatter(range(len(column_values)), column_values, s=1)
    
    # Ordered scatter plot
    plt.subplot(133)
    plt.title('Ordered scatter plot')
    plt.scatter(range(len(column_values)), column_values[idx_srt], s=1)
    
    plt.show()
    plt.close()