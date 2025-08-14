# ------------------------------
# System & Utilities
# ------------------------------
import sys
import os
import warnings
from os.path import exists
import copy

# ------------------------------
# Project-specific modules
# ------------------------------
from decision_making_analysis.cluster_replacement import cluster_replacement_utils
from decision_making_analysis.decision_making import decision_making_utils, plot_decision_making
from decision_making_analysis import free_selection, replacement, trajectory_info
from visualization.matplotlib_tools import plot_trials, monkey_heading_utils
from machine_learning.ml_methods import regression_utils, classification_utils, prep_ml_data_utils
from data_wrangling import base_processing_class
from data_wrangling import further_processing_class
from decision_making_analysis.decision_making import decision_making_helper_class
from null_behaviors import show_null_trajectory

# ------------------------------
# Scientific & Data Libraries
# ------------------------------
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------
# Scikit-learn: Model Selection
# ------------------------------
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate, KFold
)
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Scikit-learn: Metrics
# ------------------------------
from sklearn.metrics import (
    accuracy_score, hamming_loss, multilabel_confusion_matrix,
    fbeta_score, precision_score, recall_score, confusion_matrix,
    classification_report, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef
)

# ------------------------------
# Scikit-learn: Models
# ------------------------------
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

# ------------------------------
# External Gradient Boosting Libraries (optional)
# ------------------------------
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ------------------------------
# Matplotlib & Display Settings
# ------------------------------
plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class DecisionMaking(decision_making_helper_class.DecisionMakingHelper):

    def __init__(self, raw_data_folder_path=None, retrieve_monkey_data=True, time_range_of_trajectory=[-2.5, 0], num_time_points_for_trajectory=5
                 ):

        super().__init__(raw_data_folder_path=raw_data_folder_path)
        self.time_range_of_trajectory = time_range_of_trajectory
        self.gc_kwargs = {}
        self.gc_kwargs['num_time_points_for_trajectory'] = num_time_points_for_trajectory
        self.polar_plots_kwargs = {}
        self.trajectory_features = [
            'monkey_distance', 'monkey_angle_to_origin']
        if retrieve_monkey_data:
            self.get_monkey_data(include_ff_dataframe=False)

    def prepare_data_for_machine_learning(self, kind="free selection", furnish_with_trajectory_data=True, trajectory_data_kind="position", add_traj_stops=True):
        # kind can also be "replacement"
        # trajectory_data_kind can also be "velocity"
        '''
        X_all: array, containing the input features for machine learning
        y_all: array, containing the labels for machine learning
        indices: array, containing the indices of the rows in X_all and y_all
        input_features: array, containing the names of the input features
        X_all_to_plot: array, containing the input features for machine learning, for plotting
        time_all: array, containing the time for each row in X_all_to_plot
        point_index_all: array, containing the point_index for each row in X_all_to_plot
        '''

        self.data_kind = kind
        self.furnish_with_trajectory_data = furnish_with_trajectory_data
        # if using multi_class later, this will be updated to be True
        self.converting_multi_class_for_free_selection = False
        self.add_traj_stops = add_traj_stops
        self.trajectory_data_kind = trajectory_data_kind

        if kind == "free selection":
            self.X_all = self.free_selection_x_df.drop(
                columns=['point_index'], errors='ignore').values
            self.X_all_to_plot = self.free_selection_x_df_for_plotting.copy().values
            self.y_all = self.free_selection_labels.copy()
            self.indices = np.arange(len(self.free_selection_x_df))
            self.time_all = self.free_selection_time
            self.point_index_all = self.free_selection_point_index
            self.input_features = self.free_selection_x_df.columns

        elif kind == "replacement":
            self.replacement_x_df = self.changing_pursued_ff_data_diff.drop(
                ['whether_changed'], axis=1)
            self.X_all = self.replacement_x_df.values
            self.X_all_to_plot = self.replacement_inputs_for_plotting
            self.y_all = self.replacement_labels
            self.indices = np.arange(len(self.changing_pursued_ff_data_diff))
            self.time_all = self.replacement_time
            self.point_index_all = self.replacement_point_index
            self.input_features = self.replacement_x_df.columns
        elif kind is None:
            pass
        else:
            raise ValueError(
                "kind can only be 'free selection', 'replacement', or None")

        if furnish_with_trajectory_data:
            self.furnish_machine_learning_data_with_trajectory_data(
                trajectory_data_kind=trajectory_data_kind, add_traj_stops=add_traj_stops)

    def turn_y_label_into_multi_class(self, allow_multi_label=True, manual_anno_mul=None):
        self.converting_multi_class_for_free_selection = True
        if self.data_kind != "free selection":
            raise ValueError(
                "The function turn_y_label_into_multi_class is only for free selection data")

        self.y_all_original = self.y_all.copy()

        y_all = self.y_all
        if allow_multi_label:
            if manual_anno_mul is None:
                manual_anno_mul = pd.read_csv(
                    'multiff_analysis/manual_anno_multi_label.csv')
        sequence_of_obs_ff_indices = self.sequence_of_obs_ff_indices
        sequence_of_original_starting_point_index = self.chosen_rows_of_df.original_starting_point_index.values
        self.y_all, self.anno_but_not_obs_ff_indices_dict = decision_making_utils.turn_labels_into_multi_label_format(
            y_all, manual_anno_mul, self.manual_anno_long, sequence_of_obs_ff_indices, sequence_of_original_starting_point_index, allow_multi_label=allow_multi_label)

        # furnish self.anno_but_not_obs_ff_indices_dict, which might be useful for animation later
        for key, row in self.non_chosen_rows_of_df.iterrows():
            starting_point_index = int(row['starting_point_index'])
            if starting_point_index not in self.anno_but_not_obs_ff_indices_dict:
                self.anno_but_not_obs_ff_indices_dict[starting_point_index] = [
                ]
            self.anno_but_not_obs_ff_indices_dict[starting_point_index].append(
                int(row['ff_index']))

    def split_data_to_train_and_test(self, scaling_data=True, keep_whole_chunks=False, test_size=0.2):
        ''' 
        # X_train: array, containing the input features for machine learning for training
        # X_test: array, containing the input features for machine learning for testing
        # y_train: array, containing the labels for machine learning for training
        # y_test: array, containing the labels for machine learning for testing
        # indices_train: array, containing the indices of the rows in X_train and y_train
        # indices_test: array, containing the indices of the rows in X_test and y_test
        # X_test_to_plot: array, containing the input features for machine learning for testing, for plotting
        # y_test_to_plot: array, containing the labels for machine learning for testing, for plotting
        # time_to_plot: array, containing the time for each row in X_test_to_plot
        # point_index_to_plot: array, containing the point_index for each row in X_test_to_plot
        # traj_points_to_plot: array, containing the trajectory points for each row in X_test_to_plot
        '''

        self.scaling_data = scaling_data
        self.keep_whole_chunks = keep_whole_chunks
        self.test_size = test_size

        if scaling_data:
            scaler = StandardScaler()
            self.X_all_sc = scaler.fit_transform(self.X_all)  # scale data
            X_all_to_use = self.X_all_sc
        else:
            X_all_to_use = self.X_all

        if keep_whole_chunks:
            num_test_points = int(len(self.indices)*test_size)
            num_train_points = len(self.indices)-num_test_points
            # make sure that the test chunk will be a whole segment...to minimize the splitting up of the train and test chunks
            test_indice_start = np.random.randint(0, num_train_points)
            self.indices_test = self.indices[test_indice_start:
                                             test_indice_start+num_test_points]
            self.indices_train = np.setdiff1d(self.indices, self.indices_test)
            self.X_train = X_all_to_use[self.indices_train]
            self.X_test = X_all_to_use[self.indices_test]
            self.y_train = self.y_all[self.indices_train]
            self.y_test = self.y_all[self.indices_test]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test = train_test_split(
                X_all_to_use, self.y_all, self.indices, test_size=test_size)

        self.X_test_to_plot = self.X_all_to_plot[self.indices_test]
        self.y_test_to_plot = self.y_all[self.indices_test]
        self.time_to_plot = self.time_all[self.indices_test]
        self.point_index_to_plot = self.point_index_all[self.indices_test]

        if self.furnish_with_trajectory_data:
            self.traj_points_to_plot = self.traj_points[self.indices_test]
            self.traj_stops_to_plot = self.traj_stops[self.indices_test]
            # the below is for plotting, if being used
            if self.monkey_information is not None:
                self.traj_distances, self.traj_angles, self.left_end_r, self.left_end_theta, self.right_end_r, self.right_end_theta = monkey_heading_utils.find_all_mheading_components_in_polar(
                    self.monkey_information, self.time_all, self.time_range_of_trajectory, self.gc_kwargs['num_time_points_for_trajectory'])
                self.traj_distances = self.traj_distances[self.indices_test]
                self.traj_angles = self.traj_angles[self.indices_test]
                self.left_end_r = self.left_end_r[self.indices_test]
                self.left_end_theta = self.left_end_theta[self.indices_test]
                self.right_end_r = self.right_end_r[self.indices_test]
                self.right_end_theta = self.right_end_theta[self.indices_test]
        else:
            self.traj_points_to_plot = None

        print("\n input features:", self.input_features, "\n")

    def furnish_machine_learning_data_with_trajectory_data(self, trajectory_data_kind="position", add_traj_stops=True):
        '''
        # traj_points: array, containing the traj_distances and traj_angles for each row in X_all
        # trajectory_feature_names: list, containing the names of the features in traj_points
        # traj_stops: array, containing the stopping information for each row in X_all, where 1 means there has been stops in the bin and 0 means not; 
            # the number of points in each row is equal to the number of trajectory points for each row in X_all
        # trajectory_feature_names: list, containing the names of the features in traj_stops
        '''

        self.X_all, self.traj_points, self.traj_stops, self.trajectory_feature_names = trajectory_info.furnish_machine_learning_data_with_trajectory_data_func(self.X_all, self.time_all, self.monkey_information,
                                                                                                                                                               trajectory_data_kind=trajectory_data_kind, time_range_of_trajectory=self.time_range_of_trajectory, num_time_points_for_trajectory=self.gc_kwargs['num_time_points_for_trajectory'], add_traj_stops=add_traj_stops)
        self.input_features = np.concatenate(
            [self.input_features, self.trajectory_feature_names], axis=0)

    def use_machine_learning_model(self, model=None):

        self.model, self.y_pred, self.model_comparison_df = classification_utils.use_ml_model_for_classification(
            self.X_train, self.y_train, self.X_test, self.y_test, model=model,
        )
        self.y_pred = self.y_pred.ravel()

    def use_neural_network(self, n_epochs=200, batch_size=100):
        # especially useful for multi-label classification

        # but make sure the data are converted to multi-class format first
        if np.any(self.y_all > 1) or np.any(self.y_train > 1):
            self.turn_y_label_into_multi_class(allow_multi_label=False)
            self.split_data_to_train_and_test(
                scaling_data=self.scaling_data, keep_whole_chunks=self.keep_whole_chunks, test_size=self.test_size)
            print("The y label has been converted to multi-class format, and allow_multi_label was set to be False. If it needs to be True, call the method turn_y_label_into_multi_class manually.")
        self.nn_model, self.y_pred = classification_utils.use_neural_network_on_classification_func(
            self.X_train, self.y_train, self.X_test, self.y_test, n_epochs=n_epochs, batch_size=batch_size)

    def use_knn(self):
        # especially useful for multi-label classification

        if np.any(self.y_all > 1):
            self.turn_y_label_into_multi_class(allow_multi_label=True)
            self.split_data_to_train_and_test(
                scaling_data=self.scaling_data, keep_whole_chunks=self.keep_whole_chunks, test_size=self.test_size)
            print("The y label has been converted to multi-class format.")

        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        y_train = self.y_train.copy()
        y_test = self.y_test.copy()

        # Create a multi-label classifier
        classifier = MultiOutputClassifier(KNeighborsClassifier())

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = classifier.predict(X_test)

        # Calculate accuracy and Hamming loss
        accuracy = accuracy_score(y_test, y_pred)

        # In multilabel classification, this function computes subset accuracy: the set of free_selection_labels predicted for a sample must exactly match the corresponding set of free_selection_labels in y_true.
        print("Accuracy:", accuracy)
        # Hamming loss is the fraction of wrong free_selection_labels to the total number of free_selection_labels.
        print("Hamming Loss:", hamming_loss(y_test, y_pred))
        print("Precision:", precision_score(
            y_test, y_pred, average="micro", zero_division=np.nan))
        print("Recall:", recall_score(y_test, y_pred,
              average="micro", zero_division=np.nan))
        print("F2 score:", fbeta_score(y_test, y_pred,
              beta=1, average="micro", zero_division=np.nan))
        print("Multilabel confusion matrix:\n",
              multilabel_confusion_matrix(y_test, y_pred))

        self.knn_model = classifier
        self.y_pred = y_pred

    def get_pred_results_df(self):
        '''
        pred_results_df: df, containing the time, y_real, y_pred, and probability for each row in X_test
        wrong_predictions_df: df, containing the time, y_real, y_pred, and probability for each row in X_test that is wrong
        wrong_predictions: array, containing the indices of the rows in X_test that is wrong
        y_pred_prob_all: array, containing the probability of each label for each row in X_test
        y_pred_prob: array, containing the probability of the predicted label for each row in X_test
        '''

        if self.converting_multi_class_for_free_selection:
            whether_matched = np.all(self.y_test == self.y_pred, axis=1)
            self.pred_results_df = pd.DataFrame({'time': self.time_to_plot,
                                                'matched': whether_matched})
            self.wrong_predictions_df = self.pred_results_df[self.pred_results_df['matched'] == False]
            self.wrong_predictions = self.wrong_predictions_df.index.to_numpy()
        else:
            self.y_pred_prob_all = self.model.predict_proba(self.X_test)
            # take out only the probability of the predicted labels
            self.y_pred_prob = self.y_pred_prob_all[np.arange(
                len(self.y_pred)), self.y_pred]

            self.pred_results_df = pd.DataFrame({'time': self.time_to_plot,
                                                 'y_real': self.y_test,
                                                 'y_pred': self.y_pred,
                                                 'probability': self.y_pred_prob})
            self.pred_results_df['matched'] = self.pred_results_df['y_real'] == self.pred_results_df['y_pred']
            self.wrong_predictions_df = self.pred_results_df[self.pred_results_df['matched'] == False]
            self.wrong_predictions = self.wrong_predictions_df.index.to_numpy()

    def add_additional_info_to_plot(self, time_range_of_trajectory, num_time_points_for_trajectory, ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis']):
        '''
        more_ff_df: df, containing the input features for additional ff for each point_index in point_index_all, with each point_index corresponds to >= 1 row
        more_ff_inputs_df_for_plotting: df, containing the input features for additional ff for each point_index in point_index_all, for plotting, in free_selection format, 
            with each point_index corresponds to 1 row
        more_ff_inputs: array, containing the input features for additional ff for each point_index in point_index_all, for plotting
        more_traj_points: array, containing additional trajectory points (including the traj_distances and traj_angles) for each row in X_all, for plotting
        more_traj_stops: array, containing the stopping information for each trajectory_point, for plotting, where 1 means there has been stops in the bin and 0 means not;
        more_ff_inputs_to_plot: array, the part of more_ff_inputs corresponding to the test set
        more_traj_points_to_plot: array, the part of more_traj_points corresponding to the test set
        more_traj_stops_to_plot: array, the part of more_traj_stops corresponding to the test set
        '''

        self.more_ff_df, self.more_ff_inputs_df_for_plotting = cluster_replacement_utils.find_more_ff_inputs_for_plotting(
            self.point_index_all, self.sequence_of_obs_ff_indices, self.ff_dataframe, self.ff_real_position_sorted, self.monkey_information, ff_attributes=ff_attributes)
        self.more_ff_inputs = self.more_ff_inputs_df_for_plotting.values
        _, self.more_traj_points, self.more_traj_stops, _ = trajectory_info.furnish_machine_learning_data_with_trajectory_data_func(self.X_all, self.time_all, self.monkey_information,
                                                                                                                                    trajectory_data_kind=self.trajectory_data_kind, time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory, add_traj_stops=self.add_traj_stops)
        self.more_ff_inputs_to_plot = self.more_ff_inputs[self.indices_test]
        self.more_traj_points_to_plot = self.more_traj_points[self.indices_test]
        self.more_traj_stops_to_plot = self.more_traj_stops[self.indices_test]

    def prepare_to_plot_prediction_results(self, use_more_ff_inputs=False, use_more_traj_points=False, show_direction_of_monkey_on_trajectory=False):
        '''
        polar_plots_kwargs: dict, containing the additional information for plotting
        ff_inputs: array, containing the input features for plotting
        labels: array, containing the labels
        y_pred: array, containing the predicted labels
        time: array, containing the time for each row in ff_inputs
        traj_points: array, containing the trajectory points for each row in ff_inputs
        traj_stops: array, containing the stopping information for each trajectory_point, where 1 means there has been stops in the bin and 0 means not;
        mheading: dict, containing the monkey direction information for each row in ff_inputs     
        '''

        ff_inputs = self.X_test_to_plot.copy()
        time = self.pred_results_df['time'].values.copy()
        labels = self.y_test_to_plot.copy()
        y_pred = self.y_pred.copy()
        if 'probability' in self.pred_results_df.columns:
            y_prob = self.pred_results_df.probability.values.copy()
        else:
            y_prob = None

        traj_points_to_plot = None
        traj_stops_to_plot = None
        if self.traj_points_to_plot is not None:
            traj_points_to_plot = self.traj_points_to_plot.copy()
            if len(self.traj_stops) > 0:
                traj_stops_to_plot = self.traj_stops_to_plot.copy()

        more_ff_inputs_to_plot = None
        if use_more_ff_inputs:
            more_ff_inputs_to_plot = self.more_ff_inputs_to_plot.copy()

        more_traj_points_to_plot = None
        more_traj_stops_to_plot = None
        if use_more_traj_points:
            more_traj_points_to_plot = self.more_traj_points_to_plot.copy()
            if len(self.more_traj_stops_to_plot) > 0:
                more_traj_stops_to_plot = self.more_traj_stops_to_plot.copy()

        mheading = None
        if show_direction_of_monkey_on_trajectory:
            if self.furnish_with_trajectory_data is True:
                mheading = {'traj_r': self.traj_distances, 'traj_theta': self.traj_angles,
                            'left_end_r': self.left_end_r, 'left_end_theta': self.left_end_theta,
                            'right_end_r': self.right_end_r, 'right_end_theta': self.right_end_theta}
            else:
                show_direction_of_monkey_on_trajectory = False
                warnings.warn(
                    "show_direction_of_monkey_on_trajectory is set to False because furnish_with_trajectory_data was False")

        self.polar_plots_kwargs = {'ff_inputs': ff_inputs,
                                   'labels': labels,
                                   'y_pred': y_pred,
                                   'time': time,
                                   'y_prob': y_prob,
                                   'num_ff_per_row': self.num_ff_per_row,
                                   'traj_points_to_plot': traj_points_to_plot,
                                   'traj_stops_to_plot': traj_stops_to_plot,
                                   'mheading': mheading,
                                   'show_direction_of_monkey_on_trajectory': show_direction_of_monkey_on_trajectory,
                                   'more_ff_inputs_to_plot': more_ff_inputs_to_plot,
                                   'more_traj_points_to_plot': more_traj_points_to_plot,
                                   'more_traj_stops_to_plot': more_traj_stops_to_plot}

        try:
            self.polar_plots_kwargs['trajectory_features'] = self.gc_kwargs['trajectory_features']
        except KeyError:
            try:
                self.polar_plots_kwargs['trajectory_features'] = self.trajectory_features
            except KeyError:
                raise ValueError(
                    'trajectory_features is not found in gc_kwargs or self.trajectory_features')

    def plot_prediction_results(self, selected_cases=None, PlotTrials_args=None, also_show_regular_plot=False, max_plot_to_make=40, show_direction_of_monkey_on_trajectory=False, use_more_ff_inputs=False, use_more_traj_points=False):

        if selected_cases is None:
            selected_cases = np.arange(len(self.X_test_to_plot))

        all_selected_cases = selected_cases.copy()
        if also_show_regular_plot:
            if PlotTrials_args is None:
                raise ValueError(
                    "PlotTrials_args cannot be None if also_show_regular_plot is True")
        else:
            # so that when using the for loop later, it will still only call the function once
            all_selected_cases = [all_selected_cases]

        # break the cases down into case by case in order to plot the regular plot
        for selected_cases in all_selected_cases:
            if also_show_regular_plot:
                selected_cases = np.array([selected_cases])

            if self.polar_plots_kwargs is None:
                self.prepare_to_plot_prediction_results(use_more_ff_inputs=use_more_ff_inputs, use_more_traj_points=use_more_traj_points,
                                                        show_direction_of_monkey_on_trajectory=show_direction_of_monkey_on_trajectory)

            self.prepare_to_plot_prediction_results(use_more_ff_inputs=use_more_ff_inputs, use_more_traj_points=use_more_traj_points,
                                                    show_direction_of_monkey_on_trajectory=show_direction_of_monkey_on_trajectory)

            if self.converting_multi_class_for_free_selection:
                # Turn the formats into T/F for plotting
                self.polar_plots_kwargs['labels'] = self.y_test_to_plot == 1
                self.polar_plots_kwargs['y_pred'] = self.y_pred == 1
                self.polar_plots_kwargs['y_prob'] = None

            plot_decision_making.make_polar_plots_for_decision_making(**self.polar_plots_kwargs,
                                                                      selected_cases=selected_cases,
                                                                      data_kind=self.data_kind,
                                                                      max_plot_to_make=max_plot_to_make)

            if also_show_regular_plot:
                duration = self.time_range_of_trajectory + \
                    self.time_to_plot[selected_cases[0]]
                print("selected_cases", selected_cases)
                print("duration", duration)
                returned_info = plot_trials.PlotTrials(
                    duration,
                    *PlotTrials_args,
                    hitting_arena_edge_ok=True
                )
                plt.show()

    def find_and_package_arc_to_center_info_for_plotting(self, all_point_index, all_ff_index, ignore_error=True):
        self.null_arc_to_center_info_for_plotting = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(all_point_index, all_ff_index, self.monkey_information, self.ff_real_position_sorted,
                                                                                                                          ignore_error=ignore_error)


# this needs to be updated since DecisionMaking was updated
def test_dm_replacement_hyperparameters(ff_dataframe, ff_caught_T_new, ff_real_position_sorted, monkey_information, auto_annot,
                                        add_arc_info=True, arc_info_to_add=['opt_arc_curv', 'curv_diff'], add_current_curv_of_traj=True, furnish_with_trajectory_data=True, num_time_points_for_trajectory=20,
                                        ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis'], trajectory_data_kind=['position'], curvature_df=None,
                                        time_range_of_trajectory=[-0.8, 0.8], n_seconds_before_crossing_boundary=0.8, n_seconds_after_crossing_boundary=0.8,
                                        replacement_inputs_format='diff_between_old_and_new'):

    dm = DecisionMaking(raw_data_folder_path=raw_data_folder_path,
                        time_range_of_trajectory=time_range_of_trajectory, num_time_points_for_trajectory=num_time_points_for_trajectory)
    dm.manual_anno = auto_annot
    dm.separate_manual_anno()
    dm.eliminate_crossing_boundary_cases(n_seconds_before_crossing_boundary=n_seconds_before_crossing_boundary,
                                         n_seconds_after_crossing_boundary=n_seconds_after_crossing_boundary)
    dm.get_replacement_x_df(add_arc_info=add_arc_info, arc_info_to_add=arc_info_to_add, add_current_curv_of_traj=add_current_curv_of_traj,
                            curvature_df=curvature_df, ff_attributes=ff_attributes, replacement_inputs_format=replacement_inputs_format)
    dm.prepare_data_for_machine_learning(
        kind="replacement", furnish_with_trajectory_data=furnish_with_trajectory_data, trajectory_data_kind=trajectory_data_kind)
    dm.split_data_to_train_and_test(scaling_data=True)
    dm.use_machine_learning_model(model=None)

    return dm.model_comparison_df
