
from planning_analysis.plan_factors import test_vs_control_utils, test_vs_control_utils
from planning_analysis.variations_of_factors_vs_results import make_variations_utils, process_variations_utils
from planning_analysis.show_planning import show_planning_class
from planning_analysis.show_planning.get_stops_near_ff import find_stops_near_ff_utils
from planning_analysis.plan_factors import plan_factors_utils
from planning_analysis import ml_methods_utils, ml_methods_class
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class CompareYValues():