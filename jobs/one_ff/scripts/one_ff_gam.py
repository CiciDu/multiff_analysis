import sys
import os
from pathlib import Path
import argparse

# -------------------------------------------------------
# Repo path bootstrap
# -------------------------------------------------------
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break

%load_ext autoreload
%autoreload 2

import os, sys, sys
from pathlib import Path
for p in [Path.cwd()] + list(Path.cwd().parents):
    if p.name == 'Multifirefly-Project':
        os.chdir(p)
        sys.path.insert(0, str(p / 'multiff_analysis/multiff_code/methods'))
        break
    

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import one_ff_pgam_design
from neural_data_analysis.neural_analysis_tools.pgam_tools import pgam_class
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff.parameters import default_prs
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import population_analysis_utils, one_ff_data_processing

import sys
import math
import gc
import subprocess
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from scipy import linalg, interpolate
from scipy.signal import fftconvolve
from scipy.io import loadmat
from scipy import sparse
import torch
from numpy import pi
import cProfile
import pstats
import json

# Machine Learning imports
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.multivariate.cancorr import CanCorr
import statsmodels.api as sm

# Neuroscience specific imports
import neo
import rcca

# To fit gpfa
import numpy as np
from importlib import reload
from scipy.integrate import odeint
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.gpfa import GPFA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from elephant.gpfa import gpfa_core, gpfa_util

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

print("done")


%load_ext autoreload
%autoreload 2
%matplotlib inline

pd.set_option('display.max_colwidth', 200)



pgam_path = '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/external/pgam/src/'
import sys
if not pgam_path in sys.path: 
    sys.path.append(pgam_path)
    
import numpy as np
import sys
from PGAM.GAM_library import *
import PGAM.gam_data_handlers as gdh
import matplotlib.pylab as plt
import pandas as pd
from post_processing import postprocess_results
from scipy.io import savemat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_folder_path',
        default=None,
        help='Run decoding only on this raw data folder',
    )

from scipy.io import loadmat

data = loadmat('all_monkey_data/one_ff_data/sessions_python.mat',
               squeeze_me=True,
               struct_as_record=False)

sessions = data['sessions_out']



# indices
session_num = 0
trial_num = 0

# params
prs = default_prs()

# session / behaviour
session = data['sessions_out'][session_num]
behaviour = session.behaviour

# trials / stats
all_trials = behaviour.trials
all_stats = behaviour.stats
trial_ids = np.arange(len(all_trials))

trial = all_trials[trial_num]
stats = all_stats[trial_num]
pos_rel = stats.pos_rel

# continuous data
continuous = trial.continuous
print(continuous._fieldnames)

x = continuous.xmp
y = continuous.ymp
v = continuous.v
w = continuous.w
t = continuous.ts

# time step
prs.dt = round(np.mean(np.diff(t)), 5)

units = sessions[session_num].units
n_units = len(units)
 
# get unit0's trial_0 spike times
trial_neural_data = {}
for unit_id in range(len(sessions[session_num].units)):
    trial_neural_data[unit_id] = sessions[session_num].units[unit_id].trials[trial_num].tspk

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import one_ff_data_processing
from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import population_analysis_utils

covariate_names = [
    'v', 'w', 'd', 'phi',
    'r_targ', 'theta_targ',
    'eye_ver', 'eye_hor', 'move'
]

covariates_concat, trial_id_vec = population_analysis_utils.concatenate_covariates_with_trial_id(
    trials=all_trials,
    trial_indices=trial_ids,
    covariate_fn=lambda tr: one_ff_data_processing.compute_all_covariates(tr, prs.dt),
    time_window_fn=population_analysis_utils.full_time_window,
    covariate_names=covariate_names
)

covariates_concat['v'].shape

Y = np.zeros((len(trial_id_vec), n_units))
for k in range(n_units):
    spk_counts, trial_id_vec_spk = population_analysis_utils.concatenate_trials_with_trial_id(
        all_trials,
        trial_ids,
        lambda tr, tid: population_analysis_utils.bin_spikes(
            trial_neural_data[k],
            tr.continuous.ts
        ),
        population_analysis_utils.full_time_window
    )
    Y[:, k] = spk_counts

Y_smooth = population_analysis_utils.smooth_signal(Y, prs.neural_filtwidth) / prs.dt


all_events = {}
for event in ['t_targ', 't_move', 't_rew']:
    events_concat, trial_id_vec_evt = population_analysis_utils.concatenate_trials_with_trial_id(
        all_trials,
        trial_ids,
        lambda tr, tid: population_analysis_utils.event_impulse(tr, tid, event),
        population_analysis_utils.full_time_window
    )
    all_events[event] = events_concat


unit_idx = 0
sm_handler = one_ff_pgam_design.build_smooth_handler_for_unit(
    unit_idx=unit_idx,                         # <-- choose the unit you want
    covariates_concat=covariates_concat,
    covariate_names=covariate_names,
    trial_id_vec=trial_id_vec,
    Y_binned=Y,
    all_events=all_events,
    dt=prs.dt,
    tuning_covariates=covariate_names,  # or a subset
    use_cyclic=set(),                  # e.g., {'heading_angle'}
    order=4,
)


if __name__ == '__main__':
    main()
