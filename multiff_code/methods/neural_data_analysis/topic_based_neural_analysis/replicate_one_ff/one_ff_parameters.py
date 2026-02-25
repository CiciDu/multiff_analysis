"""
one_ff_parameters.py

Python defaults aligned with MATLAB `default_prs.m` for one-FF analyses.
Session-specific fields from monkey metadata are intentionally excluded.
"""

from dataclasses import dataclass, field
from typing import Dict, List

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
    # Acquisition / timing (MATLAB: fs_smr, factor_downsample, dt, etc.)
    fs_smr: float = 5000.0 / 6.0
    fs_lfp: float = 500.0
    filtwidth: int = 2
    filtsize: int = 4
    factor_downsample: int = 5
    dt: float = 0.006
    screendist: float = 32.5
    height: float = 10.0
    interoculardist: float = 3.5
    framerate: float = 60.0
    x0: float = 0.0
    y0: float = -32.5
    jitter_marker: float = 0.25
    mintrialduration: float = 0.5
    electrodespacing: float = 0.4

    # Static stimulus
    monk_startpos: np.ndarray = field(default_factory=lambda: np.array([0.0, -30.0]))
    fly_ONduration: float = 0.3

    # Behavioral analysis
    saccadeduration: float = 0.05
    mintrialsforstats: int = 50
    npermutations: int = 50
    saccade_thresh: float = 50.0
    saccade_duration: float = 0.15
    v_thresh: float = 5.0
    w_thresh: float = 3.0
    v_time2thresh: float = 0.05
    ncorrbins: int = 100
    pretrial: float = 0.5
    posttrial: float = 0.5
    presaccade: float = 0.5
    postsaccade: float = 0.5
    min_intersaccade: float = 0.1
    maxtrialduration: float = 4.0
    fixateduration: float = 0.75
    fixate_thresh: float = 4.0
    movingwin_trials: int = 10
    rewardwin: float = 65.0
    ptb_sigma: float = 1.0 / 6.0
    ptb_duration: float = 1.0
    blink_thresh: float = 50.0
    nanpadding: int = 5

    # LFP / spectrum
    lfp_filtorder: int = 4
    lfp_freqmin: float = 0.5
    lfp_freqmax: float = 75.0
    spectrum_tapers: List[int] = field(default_factory=lambda: [1, 1])
    spectrum_trialave: int = 1
    spectrum_movingwin: List[float] = field(default_factory=lambda: [1.5, 1.5])
    min_stationary: float = 0.5
    min_mobile: float = 0.5
    lfp_theta: List[float] = field(default_factory=lambda: [6.0, 12.0])
    lfp_theta_peak: float = 8.5
    lfp_beta: List[float] = field(default_factory=lambda: [12.0, 20.0])
    lfp_beta_peak: float = 18.5
    sta_window: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    duration_nanpad: float = 1.0
    phase_slidingwindow: np.ndarray = field(default_factory=lambda: np.arange(0.05, 2.01, 0.05))
    num_phasebins: int = 25

    # Event-aligned PSTH
    temporal_binwidth: float = 0.02
    spkkrnlwidth_seconds: float = 0.05
    spkkrnlwidth: float = 2.5
    spkkrnlsize: int = 25
    ts: Dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "move": np.arange(-0.5, 3.5 + 0.02, 0.02),
            "target": np.arange(-0.5, 3.5 + 0.02, 0.02),
            "stop": np.arange(-3.5, 0.5 + 0.02, 0.02),
            "reward": np.arange(-3.5, 0.5 + 0.02, 0.02),
        }
    )
    peaktimewindow: List[float] = field(default_factory=lambda: [-0.5, 0.5])
    minpeakprominence_neural: float = 2.0

    # Time-rescaling / correlograms / bootstrap
    ts_shortesttrialgroup: Dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "move": np.arange(-0.5, 1.5 + 0.02, 0.02),
            "target": np.arange(-0.5, 1.5 + 0.02, 0.02),
            "stop": np.arange(-1.5, 0.5 + 0.02, 0.02),
            "reward": np.arange(-1.5, 0.5 + 0.02, 0.02),
        }
    )
    ntrialgroups: int = 5
    duration_zeropad: float = 0.05
    corr_lag: float = 1.0
    nbootstraps: int = 100

    # Tuning settings
    tuning_nbins1d_binning: int = 20
    tuning_nbins2d_binning: List[int] = field(default_factory=lambda: [20, 20])
    tuning_nbins1d_knn: int = 100
    tuning_nbins2d_knn: List[int] = field(default_factory=lambda: [100, 100])
    tuning_kernel_nw: str = "Gaussian"
    tuning_kernel_locallinear: str = "Gaussian"
    tuning_use_binrange: bool = True

    # Bin ranges
    binrange: Dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "v": np.array([0.0, 200.0]),
            "w": np.array([-90.0, 90.0]),
            "a": np.array([-0.36, 0.36]),
            "alpha": np.array([-0.36, 0.36]),
            "r_targ": np.array([0.0, 400.0]),
            "theta_targ": np.array([-60.0, 60.0]),
            "d": np.array([0.0, 400.0]),
            "phi": np.array([-90.0, 90.0]),
            "h1": np.array([-0.36, 0.36]),
            "h2": np.array([-0.36, 0.36]),
            "eye_ver": np.array([-25.0, 0.0]),
            "eye_hor": np.array([-40.0, 40.0]),
            "veye_vel": np.array([-15.0, 5.0]),
            "heye_vel": np.array([-30.0, 30.0]),
            "phase": np.array([-np.pi, np.pi]),
            "target_ON": np.array([-0.24, 0.48]),
            "target_OFF": np.array([-0.36, 0.36]),
            "move": np.array([-0.36, 0.36]),
            "stop": np.array([-0.36, 0.36]),
            "reward": np.array([-0.36, 0.36]),
            "spikehist": np.array([0.006, 0.246]),
            # Python aliases used by parts of this repo:
            "t_targ": np.array([-0.36, 0.36]),
            "t_move": np.array([-0.36, 0.36]),
            "t_stop": np.array([-0.36, 0.36]),
            "t_rew": np.array([-0.36, 0.36]),
            "spike_hist": np.array([0.006, 0.246]),
        }
    )

    # Model fitting
    neuralfiltwidth: int = 10
    neural_filtwidth: int = 10
    nfolds: int = 5
    decodertype: str = "lineardecoder"
    lineardecoder_fitkernelwidth: bool = False
    lineardecoder_n_splits: int = 5
    lineardecoder_cv_mode: str = "blocked_time_buffered" #"group_kfold"
    lineardecoder_buffer_samples: int = 20
    lineardecoder_subsample: bool = False
    N_neurons: np.ndarray = field(default_factory=lambda: 2 ** np.arange(10))
    N_neuralsamples: int = 20

    # Traditional analyses
    hand_features: List[str] = field(
        default_factory=lambda: [
            "Finger1",
            "Finger2",
            "Finger3",
            "Finger4",
            "Wrist-down",
            "Wrist-up",
            "Hand-down",
            "Hand-up",
        ]
    )
    tuning_events: List[str] = field(default_factory=lambda: ["move", "target", "stop", "reward"])
    tuning_continuous: List[str] = field(
        default_factory=lambda: ["v", "w", "r_targ", "theta_targ", "d", "phi", "eye_ver", "eye_hor", "phase"]
    )
    tuning_method: str = "binning"

    # GAM / NNM configs
    GAM_varname: List[str] = field(
        default_factory=lambda: [
            "v",
            "w",
            "d",
            "phi",
            "r_targ",
            "theta_targ",
            "eye_ver",
            "eye_hor",
            "phase",
            "move",
            "target_OFF",
            "stop",
            "reward",
            "spikehist",
        ]
    )
    GAM_vartype: List[str] = field(
        default_factory=lambda: [
            "1D",
            "1D",
            "1D",
            "1D",
            "1D",
            "1D",
            "1D",
            "1D",
            "1Dcirc",
            "event",
            "event",
            "event",
            "event",
            "event",
        ]
    )
    GAM_basistype: List[str] = field(
        default_factory=lambda: [
            "boxcar",
            "boxcar",
            "boxcar",
            "boxcar",
            "boxcar",
            "boxcar",
            "boxcar",
            "boxcar",
            "boxcar",
            "raisedcosine",
            "raisedcosine",
            "raisedcosine",
            "raisedcosine",
            "nlraisedcosine",
        ]
    )
    GAM_linkfunc: str = "log"
    GAM_nbins: List[int] = field(default_factory=lambda: [10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20])
    GAM_lambda: List[float] = field(default_factory=lambda: [5e1] * 14)
    GAM_alpha: float = 0.05
    GAM_varchoose: List[int] = field(default_factory=lambda: [1] * 14)
    GAM_method: str = "fastbackward"

    NNM_varname: List[str] = field(default_factory=lambda: [
        "v",
        "w",
        "d",
        "phi",
        "r_targ",
        "theta_targ",
        "eye_ver",
        "eye_hor",
        "phase",
        "move",
        "target_OFF",
        "stop",
        "reward",
        "spikehist",
    ])
    NNM_vartype: List[str] = field(
        default_factory=lambda: [
            "1D",
            "1D",
            "1D",
            "1D",
            "1D",
            "1D",
            "1D",
            "1D",
            "1Dcirc",
            "event",
            "event",
            "event",
            "event",
            "event",
        ]
    )
    NNM_nbins: List[int] = field(default_factory=lambda: [10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20])
    NNM_method: str = "feedforward_nn"

    # Population analysis
    canoncorr_varname: List[str] = field(default_factory=lambda: ["v", "w", "d", "phi", "dv", "dw"])
    simulate_varname: List[str] = field(default_factory=lambda: ["v", "w", "d", "phi", "dv", "dw"])
    simulate_vartype: List[str] = field(default_factory=lambda: ["1D", "1D", "1D", "1D", "1D", "1D", "1D", "1D"])
    readout_varname: List[str] = field(
        default_factory=lambda: ["v", "w", "d", "phi", "r_targ", "theta_targ", "dv", "dw", "eye_ver", "eye_hor"]
    )
    compute_canoncorr: bool = True
    regress_popreadout: bool = True
    simulate_population: bool = False
    corr_neuronbehverr: bool = True

    # Analysis toggles
    split_trials: bool = True
    regress_behv: bool = True
    regress_eye: bool = False
    evaluate_peaks: bool = False
    compute_tuning: bool = True
    fitGAM_tuning: bool = False
    GAM_varexp: bool = False
    fitGAM_coupled: bool = False
    fitNNM: bool = False
    event_potential: bool = True
    compute_spectrum: bool = False
    analyse_theta: bool = False
    analyse_beta: bool = False
    compute_coherencyLFP: bool = False
    analyse_spikeLFPrelation: bool = False
    analyse_spikeLFPrelation_allLFPs: bool = False
    analyse_temporalphase: bool = False

    # Plotting
    binwidth_abs: float = 0.02
    binwidth_warp: float = 0.01
    trlkrnlwidth: int = 50
    maxtrls: int = 5000
    maxrewardwin: float = 400.0
    bootstrap_trl: int = 50

    # Lookups
    varlookup: Dict[str, str] = field(
        default_factory=lambda: {
            "target_ON": "t_targ",
            "target_OFF": "t_targ",
            "move": "t_move",
            "stop": "t_stop",
            "reward": "t_rew",
            "v": "lin vel",
            "w": "ang vel",
            "a": "lin acc",
            "alpha": "ang acc",
            "d": "dist moved",
            "phi": "ang turned",
            "h1": "hand vel PC1",
            "h2": "hand vel PC2",
            "r_targ": "targ dist",
            "theta_targ": "targ ang",
            "phase": "lfp phase",
            "spikehist": "spike history",
        }
    )
    unitlookup: Dict[str, str] = field(
        default_factory=lambda: {
            "target_ON": "s",
            "target_OFF": "s",
            "move": "s",
            "stop": "s",
            "reward": "s",
            "v": "cm/s",
            "w": "deg/s",
            "a": "cm/s",
            "alpha": "deg/s",
            "d": "cm",
            "phi": "deg",
            "h1": "pixels/s",
            "h2": "pixels/s",
            "r_targ": "cm",
            "theta_targ": "deg",
            "phase": "rad",
            "spikehist": "s",
        }
    )

    def __post_init__(self) -> None:
        # Keep dt consistent with acquisition parameters.
        self.dt = float(self.factor_downsample / self.fs_smr)
        # Keep sample-domain spike kernel fields consistent with temporal_binwidth.
        self.spkkrnlwidth = float(self.spkkrnlwidth_seconds / self.temporal_binwidth)
        self.spkkrnlsize = int(round(10.0 * self.spkkrnlwidth))
        # Keep plotting default aligned with analysis binwidth.
        self.binwidth_abs = float(self.temporal_binwidth)
        # Backward-compatible alias.
        self.neural_filtwidth = int(self.neuralfiltwidth)


def default_prs() -> Params:
    """Return default one-FF analysis parameters."""
    return Params()
