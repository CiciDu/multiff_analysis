from decision_making_analysis.data_enrichment import rsw_vs_rcap_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from data_wrangling import general_utils
from decision_making_analysis.ff_data_acquisition import get_missed_ff_data, free_selection
from decision_making_analysis.event_detection import get_miss_to_switch_data
from pattern_discovery import cluster_analysis
from visualization.matplotlib_tools import plot_trials
from decision_making_analysis.data_enrichment import miss_events_enricher
from data_wrangling import base_processing_class, further_processing_class
from decision_making_analysis.data_enrichment import trajectory_class, trajectory_utils
from decision_making_analysis.ff_data_acquisition import missed_ff_data_class, cluster_replacement_utils, ff_data_utils
from machine_learning.ml_methods import classification_utils, regression_utils
from null_behaviors import show_null_trajectory, curvature_utils
from visualization.matplotlib_tools import monkey_heading_utils

# ------------------------------
# Scientific & Data Libraries
# ------------------------------
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt

# ------------------------------
# Scikit-learn: Model Selection
# ------------------------------
from sklearn.model_selection import (
    train_test_split
)
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Scikit-learn: Metrics
# ------------------------------
from sklearn.metrics import (
    accuracy_score, hamming_loss, multilabel_confusion_matrix,
    fbeta_score, precision_score, recall_score
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier


class MLForDecisionMakingClass():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self.X_all_df, self.traj_points, self.traj_stops, self.trajectory_feature_names = trajectory_utils.furnish_machine_learning_data_with_trajectory_data_func(self.X_all_df, self.time_all, self.monkey_information,
                                                                                                                                                                trajectory_data_kind=trajectory_data_kind, time_range_of_trajectory=self.time_range_of_trajectory, num_time_points_for_trajectory=self.gc_kwargs['num_time_points_for_trajectory'], add_traj_stops=add_traj_stops)
        # we already have updated
        self.input_features = np.concatenate(
            [self.input_features, self.trajectory_feature_names], axis=0)
        self.X_all = self.X_all_df.values
        return self.X_all_df, self.X_all

    def use_machine_learning_model_for_classification(self, model=None):

        self.model, self.y_pred, self.model_comparison_df = classification_utils.ml_model_for_classification(
            self.X_train, self.y_train, self.X_test, self.y_test,
        )
        self.y_pred = self.y_pred.ravel()

    def use_neural_network(self, n_epochs=200, batch_size=100):
        self.nn_model, self.y_pred = classification_utils.use_neural_network_on_classification_func(
            self.X_train, self.y_train, self.X_test, self.y_test, n_epochs=n_epochs, batch_size=batch_size)

    def use_knn(self):
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

    def find_and_package_arc_to_center_info_for_plotting(self, all_point_index, all_ff_index, ignore_error=True):
        self.null_arc_to_center_info_for_plotting = show_null_trajectory.find_and_package_arc_to_center_info_for_plotting(all_point_index, all_ff_index, self.monkey_information, self.ff_real_position_sorted,
                                                                                                                          ignore_error=ignore_error)

    def get_input_data(self, num_ff_per_row=5, select_every_nth_row=1, add_arc_info=False, arc_info_to_add=['opt_arc_curv', 'curv_diff'], curvature_df=None, curv_of_traj_df=None, **kwargs):
        self.get_free_selection_x(num_ff_per_row=num_ff_per_row, select_every_nth_row=select_every_nth_row,
                                  add_arc_info=add_arc_info, arc_info_to_add=arc_info_to_add, curvature_df=curvature_df, curv_of_traj_df=curv_of_traj_df, **kwargs)

    # ===============================================================
    # Process and align current vs alternative firefly info
    # ===============================================================
    def process_current_and_alternative_ff_info(self):

        for df in [self.miss_event_cur_ff, self.miss_event_alt_ff]:
            df[['whether_changed', 'whether_intended_target']] = False

        self.num_old_ff_per_row = self.gc_kwargs['num_old_ff_per_row']
        self.num_new_ff_per_row = self.gc_kwargs['num_new_ff_per_row']

        self.miss_event_cur_ff, self.miss_event_alt_ff = (
            cluster_replacement_utils.further_process_df_related_to_cluster_replacement(
                self.miss_event_cur_ff,
                self.miss_event_alt_ff,
                num_old_ff_per_row=self.num_old_ff_per_row,
                num_new_ff_per_row=self.num_new_ff_per_row,
                selection_criterion_if_too_many_ff=self.gc_kwargs['selection_criterion_if_too_many_ff']
            )
        )

        self.miss_event_alt_ff['order'] += self.num_old_ff_per_row

        # Fill curvature placeholders
        self.miss_event_cur_ff = curvature_utils.fill_up_NAs_for_placeholders_in_columns_related_to_curvature(
            self.miss_event_cur_ff, self.relevant_curv_of_traj_df, curv_of_traj_df=self.relevant_curv_of_traj_df
        )
        self.miss_event_alt_ff = curvature_utils.fill_up_NAs_for_placeholders_in_columns_related_to_curvature(
            self.miss_event_alt_ff, self.relevant_curv_of_traj_df, curv_of_traj_df=self.relevant_curv_of_traj_df
        )

        # also fill in eventual_outcome and event_type
        self.miss_event_cur_ff = get_missed_ff_data.fill_in_eventual_outcome_and_event_type(
            self.miss_event_cur_ff)
        self.miss_event_alt_ff = get_missed_ff_data.fill_in_eventual_outcome_and_event_type(
            self.miss_event_alt_ff)

    # ===============================================================
    # Construct ML-ready input/output
    # ===============================================================

    def prepare_data_to_predict_num_stops(
        self,
        add_arc_info=False,
        add_current_curv_of_traj=False,
        ff_attributes=['ff_distance', 'ff_angle',
                       'time_since_last_vis'],
        add_num_ff_in_cluster=False,
        arc_info_to_add=['curv_diff', 'abs_curv_diff']
    ):

        # Merge current & next firefly info
        self.miss_event_cur_ff['group'] = 'Original'
        self.miss_event_alt_ff['group'] = 'Alternative'
        self.miss_to_switch_joined_ff_info = pd.concat(
            [self.miss_event_cur_ff[self.miss_event_cur_ff['eventual_outcome'] == 'switch'],
                self.miss_event_alt_ff[self.miss_event_alt_ff['eventual_outcome'] == 'switch']], axis=0
        )

        self._prepare_data_using_current_and_alternative_ff_info(
            add_arc_info=add_arc_info,
            add_current_curv_of_traj=add_current_curv_of_traj,
            ff_attributes=ff_attributes,
            add_num_ff_in_cluster=add_num_ff_in_cluster,
            arc_info_to_add=arc_info_to_add
        )

        self.num_stops = self.num_stops_df.set_index('point_index').loc[
            self.point_index_array, 'num_stops'
        ].values
        self.y_value = self.num_stops

        self.y_var_df = pd.DataFrame(
            self.num_stops.astype(int), columns=['num_stops'])
        self.y_var_df.loc[self.y_var_df['num_stops'] > 2, 'num_stops'] = 2
        # to avoid problem during classification
        self.y_var_df['num_stops'] = self.y_var_df['num_stops'] - 1

    def prepare_data_to_predict_rsw_vs_rcap(self,
                                            add_arc_info=False,
                                            add_current_curv_of_traj=False,
                                            ff_attributes=['ff_distance', 'ff_angle',
                                                           'time_since_last_vis'],
                                            add_num_ff_in_cluster=False,
                                            arc_info_to_add=['curv_diff', 'abs_curv_diff'],
                                            use_alt_ff_only=False):
        # Merge current & next firefly info
        self.miss_event_cur_ff['group'] = 'Original'
        self.miss_event_alt_ff['group'] = 'Alternative'        
        
        if use_alt_ff_only:
            self.miss_to_switch_joined_ff_info = self.miss_event_alt_ff.copy()
            self.miss_to_switch_joined_ff_info['order'] = self.miss_to_switch_joined_ff_info['order'] - self.num_old_ff_per_row
            self.num_ff_per_row = self.num_new_ff_per_row
        else:
            self.miss_to_switch_joined_ff_info = pd.concat(
                [self.miss_event_cur_ff[self.miss_event_cur_ff['event_type'].isin(['rcap', 'rsw'])],
                    self.miss_event_alt_ff[self.miss_event_alt_ff['event_type'].isin(['rcap', 'rsw'])]], axis=0
            )
            self.num_ff_per_row = self.num_old_ff_per_row + self.num_new_ff_per_row

        self._prepare_data_using_current_and_alternative_ff_info(
            add_arc_info=add_arc_info,
            add_current_curv_of_traj=add_current_curv_of_traj,
            ff_attributes=ff_attributes,
            add_num_ff_in_cluster=add_num_ff_in_cluster,
            arc_info_to_add=arc_info_to_add
        )

        self.y_var_df = rsw_vs_rcap_utils.extract_binary_event_label(
                df=self.miss_event_cur_ff,
                point_index_array=self.point_index_array
            )
        self.y_value = self.y_var_df['whether_rcap'].values

    def _prepare_data_using_current_and_alternative_ff_info(self,
                                                            add_arc_info=False,
                                                            add_current_curv_of_traj=False,
                                                            ff_attributes=['ff_distance', 'ff_angle',
                                                                           'time_since_last_vis'],
                                                            add_num_ff_in_cluster=False,
                                                            arc_info_to_add=['curv_diff', 'abs_curv_diff']):

        if add_current_curv_of_traj:
            existing_curv_feats = [
                n for n in self.all_traj_feature_names['traj_points'] if 'curv_of_traj' in n]
            if existing_curv_feats:
                add_current_curv_of_traj = False
                print(
                    "Warning: 'curv_of_traj' already present in trajectory features. Ignoring add_current_curv_of_traj.")
                
        if not hasattr(self, 'num_ff_per_row'):
            self.num_ff_per_row = self.num_old_ff_per_row + self.num_new_ff_per_row

        self.ff_attributes = ff_attributes.copy()
        self.attributes_for_plotting = [a for a in [
            'ff_distance', 'ff_angle', 'time_since_last_vis'] if a in self.ff_attributes]

        if add_arc_info:
            ff_attributes = list(set(ff_attributes) | set(arc_info_to_add))

        # Build ML design matrix
        (self.free_selection_x_df, self.free_selection_x_df_for_plotting,
         self.sequence_of_obs_ff_indices, self.point_index_array, self.pred_var) = free_selection.find_free_selection_x_from_info_of_n_ff_per_point(
            self.miss_to_switch_joined_ff_info,
            ff_attributes=ff_attributes,
            attributes_for_plotting=self.attributes_for_plotting,
            num_ff_per_row=self.num_ff_per_row
        )

        if add_current_curv_of_traj:
            curv_of_traj = self.relevant_curv_of_traj_df.set_index('point_index').loc[
                self.point_index_array, 'curv_of_traj'
            ].values
            self.free_selection_x_df['curv_of_traj'] = curv_of_traj

        if add_num_ff_in_cluster:
            self.free_selection_x_df['num_current_ff_in_cluster'] = (
                self.miss_event_cur_ff.groupby('point_index').first()[
                    'num_ff_in_cluster'].values
            )
            self.free_selection_x_df['num_nxt_ff_in_cluster'] = (
                self.miss_event_alt_ff.groupby('point_index').first()[
                    'num_ff_in_cluster'].values
            )

        # Extract labels and times
        self.free_selection_time = self.miss_to_switch_joined_ff_info.set_index('point_index').loc[
            self.point_index_array, 'time'
        ].values

    # ===============================================================
    # Prepare for ML pipeline
    # ===============================================================
    def prepare_data_for_machine_learning(self, furnish_with_trajectory_data=False,
                                          add_traj_stops=False,
                                          trajectory_data_kind='position'):
        """
        Prepares all rsw-derived inputs and labels for ML models.
        Optionally augments with trajectory points and stops.
        """


        self.time_range_of_trajectory = self.gc_kwargs['time_range_of_trajectory']
        self.num_time_points_for_trajectory = self.gc_kwargs['num_time_points_for_trajectory']

        self.furnish_with_trajectory_data = furnish_with_trajectory_data
        self.add_traj_stops = add_traj_stops
        self.trajectory_data_kind = trajectory_data_kind

        self.X_all_df = self.free_selection_x_df.drop(
            columns=['point_index'], errors='ignore')
        self.X_all = self.X_all_df.values
        self.X_all_to_plot = self.free_selection_x_df_for_plotting.copy().values
        self.indices = np.arange(len(self.free_selection_x_df))
        self.y_all = self.y_value.copy()
        self.time_all = self.free_selection_time
        self.input_features = self.free_selection_x_df.columns

        if furnish_with_trajectory_data:
            self.X_all_df, self.X_all = self.furnish_machine_learning_data_with_trajectory_data(
                trajectory_data_kind=trajectory_data_kind, add_traj_stops=add_traj_stops)
