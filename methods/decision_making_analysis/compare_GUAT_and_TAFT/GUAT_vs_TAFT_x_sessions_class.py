
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class CompareGUATandTAFTacrossSessionsClass():
    

    def __init__(self, 
                 ref_point_mode='distance', 
                 ref_point_value=-150,
                 stop_period_duration=2):
        
        self.ref_point_mode = ref_point_mode
        self.ref_point_value = ref_point_value
        self.stop_period_duration = stop_period_duration
