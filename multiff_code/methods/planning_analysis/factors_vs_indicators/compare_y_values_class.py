
from planning_analysis.factors_vs_indicators import make_variations_utils, process_variations_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from data_wrangling import specific_utils
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
                                                    use_stored_data_only=False,
                                                    **kwargs):
        self.all_ref_pooled_median_info = self._make_or_retrieve_all_ref_median_info(
            per_sess=False, use_stored_data_only=use_stored_data_only, **kwargs)
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
                                              use_stored_data_only=False,
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
                                                                                                   use_stored_data_only=use_stored_data_only,
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

    def make_or_retrieve_per_sess_perc_info(self, exists_ok=True, stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                                            ref_point_mode='distance', ref_point_value=-50, verbose=False, save_data=True,
                                            filter_heading_info_df_across_refs=False,
                                            ):
        # These two parameters (ref_point_mode, ref_point_value) are actually not important here as long as the corresponding data can be successfully retrieved,
        # since the results are the same regardless

        if exists_ok & exists(self.per_sess_perc_info_path):
            self.per_sess_perc_info = pd.read_csv(self.per_sess_perc_info_path).drop(
                ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions2(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                    heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                                                    filter_heading_info_df_across_refs=filter_heading_info_df_across_refs)
            self.per_sess_perc_info = make_variations_utils.make_per_sess_perc_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df,
                                                                                                                       self.ctrl_heading_info_df, verbose=verbose)

        self.per_sess_perc_info['monkey_name'] = self.monkey_name
        self.per_sess_perc_info['opt_arc_type'] = self.opt_arc_type
        # this doesn't matter for perc info
        self.per_sess_perc_info['curv_traj_window_before_stop'] = '[-25, 0]'

        if save_data:
            self.per_sess_perc_info.to_csv(self.per_sess_perc_info_path)
        print('Stored new per_sess_perc_info in ',
              self.per_sess_perc_info_path)

        return self.per_sess_perc_info

    def make_or_retrieve_pooled_perc_info(self, exists_ok=True, stops_near_ff_df_exists_ok=True, heading_info_df_exists_ok=True,
                                          ref_point_mode='distance', ref_point_value=-50, verbose=False, save_data=True,
                                          filter_heading_info_df_across_refs=False,
                                          use_stored_data_only=False,
                                          ):
        # These two parameters (ref_point_mode, ref_point_value) are actually not important here as long as the corresponding data can be successfully retrieved,
        # since the results are the same regardless

        if exists_ok & exists(self.pooled_perc_info_path):
            self.pooled_perc_info = pd.read_csv(self.pooled_perc_info_path).drop(
                ["Unnamed: 0", "Unnamed: 0.1"], axis=1, errors='ignore')
        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions2(ref_point_mode=ref_point_mode, ref_point_value=ref_point_value,
                                                                    heading_info_df_exists_ok=heading_info_df_exists_ok, stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                                                                    filter_heading_info_df_across_refs=filter_heading_info_df_across_refs, use_stored_data_only=use_stored_data_only)
            self.pooled_perc_info = make_variations_utils.make_pooled_perc_info_from_test_and_ctrl_heading_info_df(self.test_heading_info_df,
                                                                                                                   self.ctrl_heading_info_df, verbose=verbose)

            if save_data:
                self.pooled_perc_info.to_csv(self.pooled_perc_info_path)
            print('Stored new pooled_perc_info in ',
                  self.pooled_perc_info_path)

        self.pooled_perc_info['monkey_name'] = self.monkey_name
        self.pooled_perc_info['opt_arc_type'] = self.opt_arc_type

        return self.pooled_perc_info

    def process_all_ref_pooled_median_info_to_plot_heading_and_curv(self):
        self.all_ref_pooled_median_info_heading = process_variations_utils.make_new_df_for_plotly_comparison(
            self.all_ref_pooled_median_info)
        self.all_ref_pooled_median_info_curv = self.all_ref_pooled_median_info_heading.copy()
        self.all_ref_pooled_median_info_curv['sample_size'] = self.all_ref_pooled_median_info_curv['sample_size_for_curv']

    def process_all_ref_per_sess_median_info_to_plot_heading_and_curv(self):
        self.all_ref_per_sess_median_info_heading = process_variations_utils.make_new_df_for_plotly_comparison(
            self.all_ref_per_sess_median_info)
        self.all_ref_per_sess_median_info_curv = self.all_ref_per_sess_median_info_heading.copy()
        self.all_ref_per_sess_median_info_curv[
            'sample_size'] = self.all_ref_per_sess_median_info_curv['sample_size_for_curv']

    def _make_median_info(self,
                          kind: str = "pooled",
                          ref_point_mode: str = "time after cur ff visible",
                          ref_point_value: float = 0.1,
                          curv_traj_window_before_stop=(-25, 0),
                          exists_ok: bool = True,
                          combd_heading_df_x_sessions_exists_ok: bool = True,
                          stops_near_ff_df_exists_ok: bool = True,
                          heading_info_df_exists_ok: bool = True,
                          verbose: bool = False,
                          save_data: bool = True,
                          filter_heading_info_df_across_refs=False,
                          use_stored_data_only=False,
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
            df = df.drop(
                columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")
            setattr(self, cfg["df_attr"], df)
            print(f"Successfully retrieved {cfg['human_name']} from {path}")
            return df

        self.get_test_and_ctrl_heading_info_df_across_sessions2(
            ref_point_mode=ref_point_mode,
            ref_point_value=ref_point_value,
            curv_traj_window_before_stop=curv_traj_window_before_stop,
            heading_info_df_exists_ok=heading_info_df_exists_ok,
            stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
            save_data=save_data,
            combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
            filter_heading_info_df_across_refs=filter_heading_info_df_across_refs,
            use_stored_data_only=use_stored_data_only,
        )

        df = cfg["make_fn"](self.test_heading_info_df,
                            self.ctrl_heading_info_df, verbose=verbose)
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

    def get_test_and_ctrl_heading_info_df_across_sessions2(self, ref_point_mode='distance', ref_point_value=-150,
                                                           curv_traj_window_before_stop=[
                                                               -25, 0],
                                                           heading_info_df_exists_ok=True,
                                                           combd_heading_df_x_sessions_exists_ok=True,
                                                           stops_near_ff_df_exists_ok=True,
                                                           save_data=True,
                                                           filter_heading_info_df_across_refs=False,
                                                           use_stored_data_only=False,
                                                           **kwargs
                                                           ):
        if filter_heading_info_df_across_refs:
            self.get_test_and_ctrl_heading_info_df_across_sessions_filtered()
            self.test_heading_info_df = self.all_test_heading_info_df_filtered
            self.ctrl_heading_info_df = self.all_ctrl_heading_info_df_filtered

            def _filter_by_ref(df, mode, value):
                return df[(df["ref_point_mode"] == mode) &
                          (df["ref_point_value"] == value)].copy()

            self.test_heading_info_df = _filter_by_ref(
                self.test_heading_info_df, ref_point_mode, ref_point_value)
            print('self.test_heading_info_df.shape',
                  self.test_heading_info_df.shape)
            self.ctrl_heading_info_df = _filter_by_ref(
                self.ctrl_heading_info_df, ref_point_mode, ref_point_value)

        else:
            self.get_test_and_ctrl_heading_info_df_across_sessions(
                ref_point_mode=ref_point_mode,
                ref_point_value=ref_point_value,
                curv_traj_window_before_stop=curv_traj_window_before_stop,
                heading_info_df_exists_ok=heading_info_df_exists_ok,
                stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                save_data=save_data,
                combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
                use_stored_data_only=use_stored_data_only,
            )
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
                                use_stored_data_only=False,
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
            use_stored_data_only=use_stored_data_only,
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

    def combine_test_and_ctrl_heading_info_df_across_sessions_and_ref_point_params(
        self,
        ref_point_params_based_on_mode=None,
        list_of_curv_traj_window_before_stop=None,
        save_data=True,
        combd_heading_df_x_sessions_exists_ok=True,
        stops_near_ff_df_exists_ok=True,
        heading_info_df_exists_ok=True,
    ):
        if ref_point_params_based_on_mode is None:
            ref_point_params_based_on_mode = self.default_ref_point_params_based_on_mode
        if list_of_curv_traj_window_before_stop is None:
            list_of_curv_traj_window_before_stop = [[-25, 0]]

        test_chunks = []
        ctrl_chunks = []

        for curv_traj_window_before_stop in list_of_curv_traj_window_before_stop:
            variations_list = specific_utils.init_variations_list_func(
                ref_point_params_based_on_mode, monkey_name=self.monkey_name
            )

            if variations_list is None or len(variations_list) == 0:
                continue

            for _, row in variations_list.iterrows():
                ref_point_mode = row["ref_point_mode"]
                ref_point_value = row["ref_point_value"]

                # Prefer the called method to return DFs; if not possible, keep your side-effects.
                self.get_test_and_ctrl_heading_info_df_across_sessions(
                    ref_point_mode=ref_point_mode,
                    ref_point_value=ref_point_value,
                    curv_traj_window_before_stop=curv_traj_window_before_stop,
                    heading_info_df_exists_ok=heading_info_df_exists_ok,
                    stops_near_ff_df_exists_ok=stops_near_ff_df_exists_ok,
                    save_data=save_data,
                    combd_heading_df_x_sessions_exists_ok=combd_heading_df_x_sessions_exists_ok,
                )

                # Create labeled copies to avoid SettingWithCopy and to tag rows BEFORE concat
                test_labeled = (
                    self.test_heading_info_df
                    .assign(
                        ref_point_mode=ref_point_mode,
                        ref_point_value=ref_point_value,
                        curv_traj_window_before_stop=str(
                            curv_traj_window_before_stop),
                        monkey_name=self.monkey_name,
                        opt_arc_type=self.opt_arc_type,
                    )
                    .copy()
                )
                ctrl_labeled = (
                    self.ctrl_heading_info_df
                    .assign(
                        ref_point_mode=ref_point_mode,
                        ref_point_value=ref_point_value,
                        curv_traj_window_before_stop=str(
                            curv_traj_window_before_stop),
                        monkey_name=self.monkey_name,
                        opt_arc_type=self.opt_arc_type,
                    )
                    .copy()
                )

                test_chunks.append(test_labeled)
                ctrl_chunks.append(ctrl_labeled)

        self.all_test_heading_info_df = pd.concat(
            test_chunks, axis=0, ignore_index=True) if test_chunks else pd.DataFrame()
        self.all_ctrl_heading_info_df = pd.concat(
            ctrl_chunks, axis=0, ignore_index=True) if ctrl_chunks else pd.DataFrame()
