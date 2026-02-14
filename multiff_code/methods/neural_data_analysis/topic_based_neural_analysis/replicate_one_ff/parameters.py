"""
parameters.py

Python equivalent of default_prs.m.
Holds analysis parameters for One-FF task.

Author: you
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


plot_var_order = [
    # egocentric / motion state
    'f_v',
    'f_w',
    'f_d',
    'f_phi',

    # target-related
    'f_r_targ',
    'f_theta_targ',

    # eye position
    'f_eye_ver',
    'f_eye_hor',
    

    # behavior / action
    'g_t_move',
    'g_t_targ',
    'g_t_stop',
    'g_t_rew',

    # history
    'h_spike_history',
]

    
@dataclass
class Params:
    # =========================
    # Sampling & timing
    # =========================
    temporal_binwidth: float = 0.02       # seconds
    pretrial: float = 0.5                 # seconds
    posttrial: float = 0.5                # seconds

    # =========================
    # Neural preprocessing
    # =========================
    neural_filtwidth: int = 10            # samples (Gaussian width)

    # =========================
    # Behavioral thresholds
    # =========================
    v_thresh: float = 5.0                 # cm/s
    w_thresh: float = 3.0                 # deg/s
    rewardwin: float = 65.0               # cm

    # =========================
    # GAM configuration
    # =========================
    GAM_varname: List[str] = field(default_factory=lambda: [
        'v', 'w', 'd', 'phi',
        'r_targ', 'theta_targ',
        'eye_ver', 'eye_hor',
        'phase',
        'move', 'target_OFF', 'stop', 'reward',
        'spikehist'
    ])

    GAM_vartype: List[str] = field(default_factory=lambda: [
        '1D', '1D', '1D', '1D',
        '1D', '1D',
        '1D', '1D',
        '1Dcirc',
        'event', 'event', 'event', 'event',
        'event'
    ])

    GAM_basistype: List[str] = field(default_factory=lambda: [
        'boxcar', 'boxcar', 'boxcar', 'boxcar',
        'boxcar', 'boxcar',
        'boxcar', 'boxcar',
        'boxcar',
        'raisedcosine', 'raisedcosine',
        'raisedcosine', 'raisedcosine',
        'nlraisedcosine'
    ])

    GAM_nbins: List[int] = field(default_factory=lambda: [
        10, 10, 10, 10,
        10, 10,
        10, 10,
        10,
        20, 20, 20, 20,
        20
    ])

    GAM_lambda: List[float] = field(default_factory=lambda: [
        5e1
    ] * 14)

    GAM_alpha: float = 0.05
    GAM_linkfunc: str = 'log'

    # =========================
    # Population analysis
    # =========================
    compute_canoncorr: bool = True
    regress_popreadout: bool = True
    simulate_population: bool = False
    corr_neuronbehverr: bool = True

    # =========================
    # Decoder
    # =========================
    decodertype: str = 'lineardecoder'

    # =========================
    # Bin ranges (important!)
    # =========================
    binrange: Dict[str, np.ndarray] = field(default_factory=lambda: {
        'v': np.array([0, 200]),
        'w': np.array([-90, 90]),
        'd': np.array([0, 400]),
        'phi': np.array([-90, 90]),
        'r_targ': np.array([0, 400]),
        'theta_targ': np.array([-60, 60]),
        'eye_ver': np.array([-25, 0]),
        'eye_hor': np.array([-40, 40]),
        #'t_targ': np.array([-0.24, 0.48]),
        't_targ': np.array([-0.36, 0.36]),
        't_move': np.array([-0.36, 0.36]),
        't_rew': np.array([-0.36, 0.36]),
        't_stop': np.array([-0.36, 0.36]),
        'spike_hist': np.array([0.006, 0.246]), # note, this is causal
    })
    
    fly_ONduration: float = 0.3





def default_prs():
    """
    Return a default Params object.
    Mirrors default_prs.m behavior.
    """
    return Params()
