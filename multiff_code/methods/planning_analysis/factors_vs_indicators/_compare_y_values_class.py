
from planning_analysis.factors_vs_indicators import make_variations_utils, process_variations_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
from pathlib import Path

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class _CompareYValues:

    def __init__(self):
        pass

    def make_or_retrieve_all_ref_pooled_median_info(self,
                                                    process_info_for_plotting=True,
                                                    **kwargs):
        self.all_ref_pooled_median_info = self._make_or_retrieve_all_ref_median_info(
            per_sess=False, **kwargs)
        if process_info_for_plotting:
            self.process_all_ref_pooled_median_info_to_plot_heading_and_curv()
        return self.all_ref_pooled_median_info
                
    def make_or_retrieve_all_ref_per_sess_median_info(self,
                                                      process_info_for_plotting=True,
                                                      **kwargs):
        self.all_ref_per_sess_median_info = self._make_or_retrieve_all_ref_median_info(
            per_sess=True, **kwargs)
        if process_info_for_plotting:
            self.process_all_ref_per_sess_median_info_to_plot_heading_and_curv()
        return self.all_ref_per_sess_median_info
        
    def _make_or_retrieve_all_ref_median_info(self,
                                              per_sess=False,
                                              exists_ok=True,
                                              pooled_median_info_exists_ok=True,
                                              per_sess_median_info_exists_ok=True,
                                              ref_point_params_based_on_mode=None,
                                              list_of_curv_traj_window_before_stop=[
                                                  [-25, 0]],
                                              save_data=True,
                                              combd_heading_df_x_sessions_exists_ok=True,
                                              stops_near_ff_df_exists_ok=True,
                                              heading_info_df_exists_ok=True,
                                              ):

        df_path = self.all_ref_pooled_median_info_path if not per_sess else self.all_ref_per_sess_median_info_folder_path
        variation_func = self.make_pooled_median_info if not per_sess else self.make_per_sess_median_info

        if exists_ok & exists(df_path):
            all_info = pd.read_csv(
                df_path).drop(columns=['Unnamed: 0'])
            print('Successfully retrieved all_ref_pooled_median_info from ',
                  df_path)
        else:
            if ref_point_params_based_on_mode is None:
                ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode

            all_info = pd.DataFrame([])
            for curv_traj_window_before_stop in list_of_curv_traj_window_before_stop:
                ref_median_info = make_variations_utils.make_variations_df_across_ref_point_values(variation_func,
                                                                                                          ref_point_params_based_on_mode=ref_point_params_based_on_mode,
                                                                                                          monkey_name=self.monkey_name,
                                                                                                          variation_func_kwargs={'pooled_median_info_exists_ok': pooled_median_info_exists_ok,
                                                                                                                                 'per_sess_median_info_exists_ok': per_sess_median_info_exists_ok,
                                                                                                                                 'curv_traj_window_before_stop': curv_traj_window_before_stop,
                                                                                                                                 'save_data': save_data,
                                                                                                                                 'combd_heading_df_x_sessions_exists_ok': combd_heading_df_x_sessions_exists_ok,
                                                                                                                                 'stops_near_ff_df_exists_ok': stops_near_ff_df_exists_ok,
                                                                                                                                 'heading_info_df_exists_ok': heading_info_df_exists_ok,
                                                                                                                                 },
                                                                                                          path_to_save=None,
                                                                                                          )
                ref_median_info['curv_traj_window_before_stop'] = str(
                    curv_traj_window_before_stop)
                all_info = pd.concat(
                    [all_info, ref_median_info], axis=0)

        all_info.reset_index(drop=True, inplace=True)
        all_info['monkey_name'] = self.monkey_name
        all_info['opt_arc_type'] = self.opt_arc_type
        all_info.to_csv(df_path)
        if per_sess:
            print(
                f'Saved all_ref_per_sess_median_info_folder_path to {self.all_ref_per_sess_median_info_folder_path}')
        else:
            print(
                f'Saved all_ref_pooled_median_info_path to {df_path}')

        return all_info

    def make_or_retrieve_pooled_perc_info(self, exists_ok=True, stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                                          ref_point_mode='distance', ref_point_value=-50, verbose=False, save_data=True, process_info_for_plotting=True):
        # These two parameters (ref_point_mode, ref_point_value) are actually not important here as long as the corresponding data can be successfully retrieved,
        # since the results are the same regardless

        if exists_ok & exists(self.pooled_perc_info_path):
            self.pooled_perc_info = pd.read_csv(self.pooled_perc_info_path).drop(
                ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                   heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok)
            self.pooled_perc_info = make_variations_utils.make_pooled_perc_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df,
                                                                                                                   self.ctrl_heading_info_df, verbose=verbose)

            if save_data:
                self.pooled_perc_info.to_csv(self.pooled_perc_info_path)
            print('Stored new pooled_perc_info in ',
                  self.pooled_perc_info_path)

        self.pooled_perc_info['monkey_name'] = self.monkey_name
        self.pooled_perc_info['opt_arc_type'] = self.opt_arc_type

        if process_info_for_plotting:
            self.process_pooled_perc_info_to_plot_direction()

        return self.pooled_perc_info

    def process_all_ref_pooled_median_info_to_plot_heading_and_curv(self):
        self.all_ref_pooled_median_info_heading = process_variations_utils.make_new_df_for_plotly_comparison(
            self.all_ref_pooled_median_info)
        self.pooled_median_info_curv = self.all_ref_pooled_median_info_heading.copy()
        self.pooled_median_info_curv['sample_size'] = self.pooled_median_info_curv['sample_size_for_curv']

    def process_all_ref_per_sess_median_info_to_plot_heading_and_curv(self):
        self.all_ref_per_sess_median_info_heading = process_variations_utils.make_new_df_for_plotly_comparison(
            self.all_ref_per_sess_median_info)
        self.all_ref_per_sess_median_info_curv = self.all_ref_per_sess_median_info_heading.copy()
        self.all_ref_per_sess_median_info_curv[
            'sample_size'] = self.all_ref_per_sess_median_info_curv['sample_size_for_curv']

    def process_pooled_perc_info_to_plot_direction(self):
        self.pooled_perc_info_new = process_variations_utils.make_new_df_for_plotly_comparison(self.pooled_perc_info,
                                                                                               match_rows_based_on_ref_columns_only=False)


# The above code defines two methods in a Python class.
    # def make_pooled_median_info(self, ref_point_mode='time after cur ff visible',
    #                             ref_point_value=0.1,
    #                             curv_traj_window_before_stop=[-25, 0],
    #                             pooled_median_info_exists_ok=True,
    #                             combd_heading_df_x_sessions_exists_ok=True,
    #                             stops_near_ff_df_exists_ok=True,
    #                             heading_info_df_exists_ok=True,
    #                             verbose=False, save_data=True):

    #     df_name = find_cvn_utils.find_diff_in_curv_df_name(
    #         ref_point_mode, ref_point_value, curv_traj_window_before_stop)
    #     df_path = os.path.join(self.pooled_median_info_folder_path, df_name)
    #     if pooled_median_info_exists_ok & exists(df_path):
    #         self.pooled_median_info = pd.read_csv(df_path).drop(
    #             ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
    #         print('Successfully retrieved pooled_median_info from ', df_path)
    #     else:
    #         self.get_test_and_ctrl_heading_info_df_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
    #                                                                curv_traj_window_before_stop=curv_traj_window_before_stop,
    #                                                                heading_info_df_exists_ok=heading_info_df_exists_ok,
    #                                                                stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data,
    #                                                                combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok)
    #         self.pooled_median_info = make_variations_utils.make_pooled_median_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df,
    #                                                                                                                    self.ctrl_heading_info_df, verbose=verbose)
    #         self.pooled_median_info['ref_point_mode'] = ref_point_mode
    #         self.pooled_median_info['ref_point_value'] = ref_point_value
    #         time_calibration = {'ref_point_mode': ref_point_mode,
    #                             'ref_point_value': ref_point_value, 'monkey_name': self.monkey_name}
    #         self.pooled_median_info.attrs.update(time_calibration)
    #         os.makedirs(self.pooled_median_info_folder_path, exist_ok=True)
    #         self.pooled_median_info.to_csv(df_path)
    #         print('Stored new pooled_median_info in ',
    #               self.pooled_median_info_folder_path)
    #     return self.pooled_median_info

    # def make_per_sess_median_info(self, ref_point_mode='time after cur ff visible',
    #                               ref_point_value=0.1,
    #                               curv_traj_window_before_stop=[-25, 0],
    #                               per_sess_median_info_exists_ok=True,
    #                               combd_heading_df_x_sessions_exists_ok=True,
    #                               stops_near_ff_df_exists_ok=True,
    #                               heading_info_df_exists_ok=True,
    #                               verbose=False, save_data=True,
    #                               **kwargs):

    #     df_name = find_cvn_utils.find_diff_in_curv_df_name(
    #         ref_point_mode, ref_point_value, curv_traj_window_before_stop)
    #     df_path = os.path.join(self.per_sess_median_info_folder_path, df_name)
    #     if per_sess_median_info_exists_ok & exists(df_path):
    #         self.per_sess_median_info = pd.read_csv(df_path).drop(
    #             ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
    #         print('Successfully retrieved per_sess_median_info from ', df_path)
    #     else:
    #         self.get_test_and_ctrl_heading_info_df_across_sessions(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
    #                                                                curv_traj_window_before_stop=curv_traj_window_before_stop,
    #                                                                heading_info_df_exists_ok=heading_info_df_exists_ok,
    #                                                                stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok, save_data=save_data,
    #                                                                combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok)
    #         self.per_sess_median_info = make_variations_utils.make_per_sess_median_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df,
    #                                                                                                                        self.ctrl_heading_info_df, verbose=verbose)
    #         self.per_sess_median_info['ref_point_mode'] = ref_point_mode
    #         self.per_sess_median_info['ref_point_value'] = ref_point_value
    #         time_calibration = {'ref_point_mode': ref_point_mode,
    #                             'ref_point_value': ref_point_value, 'monkey_name': self.monkey_name}
    #         self.per_sess_median_info.attrs.update(time_calibration)
    #         os.makedirs(self.per_sess_median_info_folder_path, exist_ok=True)
    #         self.per_sess_median_info.to_csv(df_path)
    #         print('Stored new per_sess_median_info in ',
    #               self.per_sess_median_info_folder_path)
    #     return self.per_sess_median_info



    def _ensure_heading_info(self,
                            ref_point_mode,
                            ref_point_value,
                            curv_traj_window_before_stop,
                            combd_heading_df_x_sessions_exists_ok,
                            stops_near_ff_df_exists_ok,
                            heading_info_df_exists_ok,
                            save_data):
        """Make sure test/ctrl heading info DataFrames exist on self."""
        self.get_test_and_ctrl_heading_info_df_across_sessions(
            ref_point_mode=ref_point_mode,
            ref_point_value=ref_point_value,
            curv_traj_window_before_stop=curv_traj_window_before_stop,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            save_data=save_data,
            combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
        )

    def _make_median_info(self,
                        kind: str = "pooled",
                        ref_point_mode: str = "time after cur ff visible",
                        ref_point_value: float = 0.1,
                        curv_traj_window_before_stop = (-25, 0),
                        exists_ok: bool = True,
                        combd_heading_df_x_sessions_exists_ok: bool = True,
                        stops_near_ff_df_exists_ok: bool = True,
                        heading_info_df_exists_ok: bool = True,
                        verbose: bool = False,
                        save_data: bool = True,
                        **kwargs):
        """
        Unified builder for median-info DataFrames.

        kind: 'pooled' or 'per_sess'
        """
        config = {
            "pooled": {
                "folder_attr": "pooled_median_info_folder_path",
                "df_attr": "pooled_median_info",
                "make_fn": getattr(
                    make_variations_utils,
                    "make_pooled_median_info_from_test_and_ctrl_heading_info_df"
                ),
                "human_name": "pooled_median_info",
            },
            "per_sess": {
                "folder_attr": "per_sess_median_info_folder_path",
                "df_attr": "per_sess_median_info",
                "make_fn": getattr(
                    make_variations_utils,
                    "make_per_sess_median_info_from_test_and_ctrl_heading_info_df"
                ),
                "human_name": "per_sess_median_info",
            },
        }
        if kind not in config:
            raise ValueError("kind must be 'pooled' or 'per_sess'")

        cfg = config[kind]

        df_name = find_cvn_utils.find_diff_in_curv_df_name(
            ref_point_mode, ref_point_value, curv_traj_window_before_stop
        )
        folder = getattr(self, cfg["folder_attr"])
        path = Path(folder) / df_name

        if exists_ok and path.exists():
            df = pd.read_csv(path)
            df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")
            setattr(self, cfg["df_attr"], df)
            print(f"Successfully retrieved {cfg['human_name']} from {path}")
            return df

        # Build prerequisites then compute
        self._ensure_heading_info(
            ref_point_mode=ref_point_mode,
            ref_point_value=ref_point_value,
            curv_traj_window_before_stop=curv_traj_window_before_stop,
            combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            save_data=save_data,
        )

        df = cfg["make_fn"](self.test_heading_info_df, self.ctrl_heading_info_df, verbose=verbose)
        df["ref_point_mode"] = ref_point_mode
        df["ref_point_value"] = ref_point_value
        df.attrs.update({
            "ref_point_mode": ref_point_mode,
            "ref_point_value": ref_point_value,
            "monkey_name": getattr(self, "monkey_name", None),
        })

        Path(folder).mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        setattr(self, cfg["df_attr"], df)
        print(f"Stored new {cfg['human_name']} in {folder}")
        return df

    # --- Thin wrappers for backward compatibility ---

    def make_pooled_median_info(self,
                                ref_point_mode='time after cur ff visible',
                                ref_point_value=0.1,
                                curv_traj_window_before_stop=(-25, 0),
                                pooled_median_info_exists_ok=True,
                                combd_heading_df_x_sessions_exists_ok=True,
                                stops_near_ff_df_exists_ok=True,
                                heading_info_df_exists_ok=True,
                                verbose=False, save_data=True,
                                **kwargs
                                ):
        return self._make_median_info(
            kind="pooled",
            ref_point_mode=ref_point_mode,
            ref_point_value=ref_point_value,
            curv_traj_window_before_stop=curv_traj_window_before_stop,
            exists_ok=pooled_median_info_exists_ok,
            combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            verbose=verbose,
            save_data=save_data,
        )

    def make_per_sess_median_info(self,
                                ref_point_mode='time after cur ff visible',
                                ref_point_value=0.1,
                                curv_traj_window_before_stop=(-25, 0),
                                per_sess_median_info_exists_ok=True,
                                combd_heading_df_x_sessions_exists_ok=True,
                                stops_near_ff_df_exists_ok=True,
                                heading_info_df_exists_ok=True,
                                verbose=False, save_data=True,
                                **kwargs):
        return self._make_median_info(
            kind="per_sess",
            ref_point_mode=ref_point_mode,
            ref_point_value=ref_point_value,
            curv_traj_window_before_stop=curv_traj_window_before_stop,
            exists_ok=per_sess_median_info_exists_ok,
            combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            verbose=verbose,
            save_data=save_data,
            **kwargs,
        )
