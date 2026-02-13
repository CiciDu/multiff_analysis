import numpy as np
from scipy.io import loadmat
import pandas as pd

from neural_data_analysis.topic_based_neural_analysis.replicate_one_ff import (
    one_ff_data_processing,
    population_analysis_utils,
)

class OneFFSessionData:
    def __init__(self, mat_path, prs, session_num=0):
        """
        Wrapper for loading and processing one-firefly session data.

        Parameters
        ----------
        mat_path : str
            Path to sessions_python.mat
        prs : object
            Parameter struct (e.g. from default_prs())
        session_num : int
            Which session to load
        """
        self.mat_path = mat_path
        self.prs = prs
        self.session_num = session_num

        self._load_data()
        self._load_session()
        self._set_time_step()

    # ------------------
    # Loading
    # ------------------
    def _load_data(self):
        data = loadmat(
            self.mat_path,
            squeeze_me=True,
            struct_as_record=False
        )
        self.sessions = data['sessions_out']

    def _load_session(self):
        self.session = self.sessions[self.session_num]
        self.behaviour = self.session.behaviour
        self.trlindx = self.session.trlindx
        self.sel_trials = self.trlindx.nonzero()[0]

        self.all_trials = self.behaviour.trials
        self.all_stats = self.behaviour.stats

        self.units = self.session.units
        self.n_units = len(self.units)

    def _set_time_step(self):
        """
        Infer dt from the first trial.
        """
        t = self.all_trials[0].continuous.ts
        self.prs.dt = round(np.mean(np.diff(t)), 3)

    # ------------------
    # Trial-level access
    # ------------------
    def get_trial(self, trial_num):
        trial = self.all_trials[trial_num]
        stats = self.all_stats[trial_num]

        continuous = trial.continuous

        return {
            'trial': trial,
            'stats': stats,
            'pos_rel': stats.pos_rel,
            'x': continuous.xmp,
            'y': continuous.ymp,
            'v': continuous.v,
            'w': continuous.w,
            't': continuous.ts,
        }

    def get_trial_spike_times(self, trial_num):
        """
        Returns a dict: unit_id -> spike times for a given trial.
        """
        trial_neural_data = {}
        for unit_id in range(self.n_units):
            trial_neural_data[unit_id] = (
                self.units[unit_id].trials[trial_num].tspk
            )
        return trial_neural_data

    # ------------------
    # Population-level processing
    # ------------------
    def compute_covariates(self, covariate_names):
        """
        Concatenate covariates across trials.
        """
        covariates_concat, trial_id_vec = (
            population_analysis_utils.concatenate_covariates_with_trial_id(
                trials=self.all_trials,
                trial_indices=self.sel_trials,
                covariate_fn=lambda tr: one_ff_data_processing.compute_all_covariates(
                    tr, self.prs.dt
                ),
                time_window_fn=population_analysis_utils.full_time_window,
                covariate_names=covariate_names
            )
        )

        self.covariates = covariates_concat
        self.covariate_trial_ids = trial_id_vec
        self.data_df = pd.DataFrame(self.covariates, columns=covariate_names)

    def compute_spike_counts(self):
        """
        Bin spikes for all units and concatenate across trials.
        """
        Y = np.zeros((len(self.covariate_trial_ids), self.n_units))

        for k in range(self.n_units):
            spk_counts, _ = population_analysis_utils.concatenate_trials_with_trial_id(
                self.all_trials,
                self.sel_trials,
                lambda tr, tid: population_analysis_utils.bin_spikes(
                    self.units[k].trials[tid].tspk,
                    tr.continuous.ts
                ),
                population_analysis_utils.full_time_window
            )
            Y[:, k] = spk_counts

        self.Y = Y


    def smooth_spikes(self):
        """
        Smooth spike counts using prs.neural_filtwidth.
        """
        Y_smooth = (
            population_analysis_utils.smooth_signal(
                self.Y, self.prs.neural_filtwidth
            ) / self.prs.dt
        )
        self.Y_smooth = Y_smooth


    def compute_events(self, event_names=('t_move', 't_stop', 't_targ', 't_rew')):
        """
        Concatenate event impulse vectors across trials.
        For t_targ, adds prs.fly_ONduration to mark target offset time.
        """
        all_events = {}

        for event in event_names:
            # Add fly_ONduration offset for t_targ events
            offset = self.prs.fly_ONduration if event == 't_targ' else 0.0
            
            events_concat, _ = population_analysis_utils.concatenate_trials_with_trial_id(
                self.all_trials,
                self.sel_trials,
                lambda tr, tid, ev=event, off=offset: population_analysis_utils.event_impulse(
                    tr, tid, ev, offset=off
                ),
                population_analysis_utils.full_time_window
            )
            all_events[event] = events_concat

        self.events = all_events


    def get_binned_spikes_df(self):
        return pd.DataFrame(self.Y, columns=np.arange(self.n_units))
