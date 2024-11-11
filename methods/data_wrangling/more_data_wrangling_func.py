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



def reorganize_data_into_chunks(monkey_information):

    prev_speed = 0
    chunk_counter = 0
    chunk_numbers = [] # Each time point has a number corresponding to the index of the chunk it belongs to
    new_chunk_indices = [0] # w[i] shows the index of the time point at which the i-th chunk starts

    # for each time point
    for i in range(len(monkey_information['monkey_speed'])):
        speed = monkey_information['monkey_speed'].values[i]
        # if the speed is above half of the full speed (100 cm/s) and if the previous speed is below half of the full speed
        if (speed > 100) & (prev_speed <= 100):
            # start a new chunk
            chunk_counter += 1
            new_chunk_indices.append(i)
        chunk_numbers.append(chunk_counter)
        prev_speed = speed

    chunk_numbers = np.array(chunk_numbers)
    new_chunk_indices = np.array(new_chunk_indices)

    return chunk_numbers, new_chunk_indices

