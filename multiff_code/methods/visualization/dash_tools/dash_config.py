"""
Shared configuration module for Dash visualization tools.
Centralizes matplotlib, pandas, and other configuration settings.
"""

import os
import sys
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


def configure_plotting_environment():
    """
    Configure the plotting environment with standard settings.
    This should be called once at the beginning of any visualization module.
    """
    # Matplotlib configuration
    plt.rcParams["animation.html"] = "html5"
    rc('animation', html='jshtml')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['animation.embed_limit'] = 2**128

    # Environment variables
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Pandas configuration
    pd.set_option('display.float_format', lambda x: '%.5f' % x)

    # NumPy configuration
    np.set_printoptions(suppress=True)


# Auto-configure when module is imported
configure_plotting_environment()

# Export commonly used constants
DEFAULT_PORT = 8045
DEFAULT_EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Common styling constants
CHECKLIST_STYLE = {
    'width': '50%',
    'padding': '0px 10px 10px 10px',
    'margin': '0 0 10px 0'
}

ALL_PLOTS_CHECKLIST_STYLE = {
    **CHECKLIST_STYLE,
    'background-color': '#F9F99A'
}

MONKEY_PLOT_CHECKLIST_STYLE = {
    **CHECKLIST_STYLE,
    'background-color': '#ADD8E6'
}

REFRESH_BUTTON_STYLE = {
    'margin': '10px 10px 10px 10px',
    'background-color': '#FFC0CB'
}
