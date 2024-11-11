import sys
if not '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods' in sys.path:
    sys.path.append('/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods')

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



plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)





# calculate loadings
def calculate_cca_loadings(features, variates):
    num_features = features.shape[1]
    num_variates = variates.shape[1]
    loadings = np.zeros((num_features, num_variates))
    # for each feature
    for feature in range(num_features):
        # calculate the correlation between feature and each of the variate
        for variate in range(num_variates):
            loadings[feature, variate] = np.corrcoef(features[:, feature], variates[:, variate])[1][0]

        # store into a dataframe
    return loadings





# Function to get unique feature categories based on the given parameters
def get_unique_feature_category(loading_squared, num_variates, keep_one_value_for_each_feature, max_features_to_show_per_plot):
    unique_feature_category = np.array([])
    for variate in range(num_variates):
        # If the flag is set to keep one value for each feature
        if keep_one_value_for_each_feature:
            # Sort the squared loadings, group by feature category, and keep the first (max) value
            loading_subset = loading_squared.sort_values(variate, ascending=False, key=abs).groupby('feature_category').first().reset_index(drop=False)
        else:
            # Otherwise, just copy the squared loadings
            loading_subset = loading_squared.copy()
        # Sort the subset by the variate and keep the top features up to the max limit
        loading_subset = loading_subset.sort_values(by=variate, ascending=False, key=abs).iloc[:max_features_to_show_per_plot]
        # Update the unique feature categories
        unique_feature_category = np.unique(np.concatenate([unique_feature_category, loading_subset.feature_category]))
    # Log the number of unique feature categories included in the plot
    logging.info(f"{len(unique_feature_category)} out of {len(loading_squared.feature_category.unique())} feature categories are included in the plot")
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





# Function to make a series of bar plots of ranked loadings
def make_a_series_of_barplots_of_ranked_loadings(loading_squared, canon_corr, num_variates, 
                                                 keep_one_value_for_each_feature = True, 
                                                 max_features_to_show_per_plot = 20, 
                                                 horizontal_bars=True):
    # Get the unique feature categories
    unique_feature_category = get_unique_feature_category(loading_squared, num_variates, keep_one_value_for_each_feature, max_features_to_show_per_plot)
    # Generate a color dictionary for the unique feature categories
    color_dict = get_color_dict(unique_feature_category)

    # Iterate over the number of variates
    for variate in range(num_variates):
        # If the flag is set to keep one value for each feature
        if keep_one_value_for_each_feature:
            # Sort the squared loadings, group by feature category, and keep the first (max) value
            loading_subset = loading_squared.sort_values(variate, ascending=False, key=abs).groupby('feature_category').first().reset_index(drop=False)
        else:
            # Otherwise, just copy the squared loadings
            loading_subset = loading_squared.copy()
        # Sort the subset by the variate and keep the top features up to the max limit
        loading_subset = loading_subset.sort_values(by=variate, ascending=False, key=abs).iloc[:max_features_to_show_per_plot]

        # Create a new plot
        fig, ax = plt.subplots(figsize=(10, 8))

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
        