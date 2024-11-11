import sys
from decision_making_analysis.decision_making import plot_decision_making

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)




def make_one_polar_plot_for_cluster_replacement(num_old_ff_per_row, predict_num_stops=False, 
                                                real_number_over_label_offset=1,
                                                **current_polar_plot_kargs):
                                                
    
    ff_input = current_polar_plot_kargs['ff_input']
    real_label = current_polar_plot_kargs['real_label']
    predicted_label = current_polar_plot_kargs['predicted_label']

    ax, markers, marker_labels = plot_decision_making.make_one_polar_plot(**current_polar_plot_kargs)
    

    marker1 = ax.scatter(ff_input[:num_old_ff_per_row,1], ff_input[:num_old_ff_per_row,0], 220, facecolor="white", edgecolor='red', alpha=1, zorder=1)
    markers.insert(0, marker1)
    marker_labels.insert(0, 'Possible current targets')
    #ax.annotate('Current', xy=(ff_input[0,1], ff_input[0,0]), xytext=(ff_input[0,1], ff_input[0,0]+30), color='red', fontsize=11)

    marker2 = ax.scatter(ff_input[num_old_ff_per_row:,1], ff_input[num_old_ff_per_row:,0], 150, facecolor="white", edgecolor='blue', alpha=1, zorder=1)
    markers.insert(0, marker2)
    marker_labels.insert(0, 'Possible alternative targets')
    #ax.annotate('Alternative', xy=(ff_input[num_old_ff_per_row,1], ff_input[num_old_ff_per_row,0]), xytext=(ff_input[num_old_ff_per_row,1], ff_input[num_old_ff_per_row,0]+40), color='blue', fontsize=11)

    # Also annotate the original ff if the data kind of replacement
    print('\n')
    if predict_num_stops == False:
        # Annotate the ff that the monkey pursues in reality
        if real_label == 0:
            real_label_one_indice = 0
        else:
            real_label_one_indice = num_old_ff_per_row
        # Annotate the ff that the monkey pursues based on prediction
        if predicted_label == 0:
            predicted_label_one_indice = 0
        else:
            predicted_label_one_indice = num_old_ff_per_row
        # If the real and the predicted labels are the same, they will be annotated with the same color
        if real_label == predicted_label:
            predicted_label_color = 'green'
        else:
            predicted_label_color = 'purple'
        ax.annotate('Real', xy=(ff_input[real_label_one_indice,1], ff_input[real_label_one_indice,0]), 
                    xytext=(ff_input[real_label_one_indice,1], max(0, ff_input[real_label_one_indice,0]-20)), color='green', fontsize=10)
        ax.annotate('Predicted', xy=(ff_input[predicted_label_one_indice,1], ff_input[predicted_label_one_indice,0]), 
                    xytext=(ff_input[predicted_label_one_indice,1], max(0, ff_input[predicted_label_one_indice,0]-40)), color=predicted_label_color, fontsize=10)
    else:
        # If the real and the predicted labels are the same, they will be annotated with the same color
        if real_label == predicted_label:
            predicted_label_color = 'green'
        else:
            predicted_label_color = 'purple'
        # annotated real_label at the bottom of the plot
        ax.annotate('Real: '+str(real_label + real_number_over_label_offset), xy=(pi, 200), xytext=(pi-0.27, 200), color='green', fontsize=10)
        ax.annotate('Predicted: '+str(predicted_label + real_number_over_label_offset), xy=(pi, 200), xytext=(pi-0.27, 230), color=predicted_label_color, fontsize=10)


    ax.legend(markers, marker_labels, scatterpoints=1, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return ax




def make_one_polar_plot_for_GUAT(num_old_ff_per_row, **current_polar_plot_kargs):

    ff_input = current_polar_plot_kargs['ff_input']
    real_label = current_polar_plot_kargs['real_label']
    predicted_label = current_polar_plot_kargs['predicted_label']


    ax, markers, marker_labels = plot_decision_making.make_one_polar_plot(**current_polar_plot_kargs)
    
    marker1 = ax.scatter(ff_input[:num_old_ff_per_row,1], ff_input[:num_old_ff_per_row,0], 220, facecolor="white", edgecolor='red', alpha=1, zorder=1)
    markers.insert(0, marker1)
    marker_labels.insert(0, 'Original')
    ax.annotate('Original', xy=(ff_input[0,1], ff_input[0,0]), xytext=(ff_input[0,1], ff_input[0,0]+30), color='red', fontsize=11)

    marker2 = ax.scatter(ff_input[num_old_ff_per_row:,1], ff_input[num_old_ff_per_row:,0], 150, facecolor="white", edgecolor='blue', alpha=1, zorder=1)
    markers.insert(0, marker2)
    marker_labels.insert(0, 'Alternative')
    ax.annotate('Alternative', xy=(ff_input[num_old_ff_per_row,1], ff_input[num_old_ff_per_row,0]), xytext=(ff_input[num_old_ff_per_row,1], ff_input[num_old_ff_per_row,0]+40), color='blue', fontsize=11)

    # Also annotate the original ff if the data kind of replacement
    print('\n')
    # If the real and the predicted labels are the same, they will be annotated with the same color
    if real_label == predicted_label:
        predicted_label_color = 'green'
    else:
        predicted_label_color = 'purple'
    # annotated real_label at the bottom of the plot
    ax.annotate('Real'+str(real_label), xy=(pi, 200), xytext=(pi-0.3, 200), color='green', fontsize=10)
    ax.annotate('Predicted'+str(predicted_label), xy=(pi, 200), xytext=(pi-0.3, 220), color=predicted_label_color, fontsize=10)


    ax.legend(markers, marker_labels, scatterpoints=1, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return ax

        




