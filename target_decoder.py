from non_behavioral_analysis.neural_data_analysis.model_neural_data.cca_methods import cca_class, cca_plotting
from non_behavioral_analysis.neural_data_analysis.decode_targets.decode_target_class import DecodeTargetClass
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project path
project_folder = '/Users/dusiyi/Documents/Multifirefly-Project'
sys.path.append(os.path.join(project_folder, 'multiff_analysis', 'methods'))


class TargetDecoder:
    """
    A comprehensive class for decoding monkey's representation of targets from neural data.

    This class integrates multiple decoding approaches:
    1. Canonical Correlation Analysis (CCA)
    2. Gaussian Process Factor Analysis (GPFA)
    3. Machine Learning approaches (RF, SVM, Neural Networks)
    4. Dimensionality reduction techniques (PCA)

    The decoder can predict various target properties including:
    - Target position (x, y coordinates)
    - Target distance and angle
    - Target visibility
    - Target approach behavior
    """

    def __init__(self, raw_data_folder_path, bin_width=0.02, window_width=0.05):
        """
        Initialize the target decoder.

        Parameters:
        -----------
        raw_data_folder_path : str
            Path to the raw monkey data folder
        bin_width : float
            Width of time bins for neural data (in seconds)
        window_width : float
            Width of sliding window for neural data (in seconds)
        """
        self.raw_data_folder_path = raw_data_folder_path
        self.bin_width = bin_width
        self.window_width = window_width

        # Initialize the decode target class
        self.decode_target = DecodeTargetClass(
            raw_data_folder_path=raw_data_folder_path,
            bin_width=bin_width,
            window_width=window_width
        )

        # Storage for models and results
        self.models = {}
        self.results = {}
        self.scalers = {}

        # Data containers
        self.neural_data = None
        self.behavioral_data = None
        self.target_data = None

    def load_and_prepare_data(self, exists_ok=True, use_lags=True):
        """
        Prepare neural and behavioral data for decoding.

        Parameters:
        -----------
        exists_ok : bool
            Whether to load existing processed data if available
        use_lags : bool
            Whether to include lagged features
        """
        print("Preparing neural and behavioral data...")

        # # Get the basic data
        # self.decode_target.streamline_making_behav_and_neural_data(
        #     exists_ok=exists_ok)

        # Get X (neural) and Y (behavioral) variables
        self.decode_target.get_x_and_y_var(exists_ok=exists_ok)
        self.decode_target.reduce_y_var_lags(exists_ok=exists_ok)
        self.decode_target._make_or_retrieve_target_df(exists_ok=exists_ok)

        if use_lags:
            # Use lagged data for better temporal modeling
            self.neural_data = self.decode_target.x_var_lags.drop(
                columns=['bin']) if 'bin' in self.decode_target.x_var_lags.columns else self.decode_target.x_var_lags
            self.behavioral_data = self.decode_target.y_var_lags_reduced
        else:
            # Use non-lagged data
            self.neural_data = self.decode_target.x_var
            self.behavioral_data = self.decode_target.y_var_reduced

        # Extract target-specific features
        self._extract_target_data()

        print(f"Neural data shape: {self.neural_data.shape}")
        print(f"Behavioral data shape: {self.behavioral_data.shape}")
        print(f"Target features shape: {self.target_data.shape}")

    def _extract_target_data(self):
        # """Extract target-related features from behavioral data."""
        # target_columns = [col for col in self.behavioral_data.columns
        #                   if any(keyword in col.lower() for keyword in
        #                          ['target_x', 'target_y', 'target_distance', 'target_angle',
        #                          'target_visible', 'target_rel', 'time_since_target'])]

        # if len(target_columns) == 0:
        #     print(
        #         "Warning: No target-specific columns found. Using all behavioral features.")
        #     self.target_data = self.behavioral_data.copy()
        # else:
        #     self.target_data = self.behavioral_data[target_columns].copy()

        self.target_data = self.behavioral_data.copy()

        # Remove any columns with all NaN values
        self.target_data = self.target_data.dropna(axis=1, how='all')

        # Fill remaining NaN values with forward fill then backward fill
        self.target_data = self.target_data.fillna(
            method='ffill').fillna(method='bfill')

    def decode_with_cca(self, n_components=10, use_lags=True):
        """
        Decode target representation using Canonical Correlation Analysis.

        Parameters:
        -----------
        n_components : int
            Number of canonical components to extract
        use_lags : bool
            Whether to use lagged features

        Returns:
        --------
        dict : CCA results including canonical correlations and loadings
        """
        print("Performing CCA-based decoding...")

        # Prepare data
        X = self.neural_data.fillna(0)  # Neural data
        Y = self.target_data.fillna(0)  # Target features

        # Initialize CCA
        cca = cca_class.CCAclass(X1=X, X2=Y, lagging_included=use_lags)

        # Conduct CCA
        cca.conduct_cca(n_components=n_components, plot_correlations=True)

        # Store results
        self.models['cca'] = cca
        self.results['cca'] = cca.results

        print(
            f"CCA completed. Top 3 canonical correlations: {cca.canon_corr[:3]}")

        return self.results['cca']

    def plot_cca_results(self):
        """Plot CCA results including canonical correlations and loadings."""
        if 'cca' not in self.results:
            print("No CCA results available. Run decode_with_cca first.")
            return

        cca_plotting.plot_cca_results(self.results['cca'])

    def plot_neural_loadings(self, max_components=5, ax=None):

        max_components = min(
            max_components, self.results['cca']['X2_loading'].shape[1])

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))

        # Neural loadings heatmap
        im1 = ax.imshow(self.results['cca']['X1_loading'][:, :max_components].T,
                        aspect='auto', cmap='RdBu_r')
        ax.set_title('Neural Loadings')
        ax.set_xlabel('Neurons')
        ax.set_ylabel('Components')
        # Set x tick labels to show neuron names
        neural_columns = self.neural_data.columns.tolist()
        ax.set_xticks(range(len(neural_columns)))
        ax.set_xticklabels(neural_columns, rotation=90, ha='right')
        plt.colorbar(im1, ax=ax)

    def plot_behav_loadings(self, max_components=10, ax=None):

        max_components = min(
            max_components, self.results['cca']['X2_loading'].shape[1])

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # Behavioral loadings heatmap
        im2 = ax.imshow(self.results['cca']['X2_loading'][:, :max_components].T,
                        aspect='auto', cmap='RdBu_r')
        ax.set_title('Behavioral Loadings')
        ax.set_xlabel('Behavioral Features')
        ax.set_ylabel('Components')
        # Set x tick labels to show behavioral feature names
        behavioral_columns = self.target_data.columns.tolist()
        ax.set_xticks(range(len(behavioral_columns)))
        ax.set_xticklabels(behavioral_columns, rotation=90, ha='right')
        plt.colorbar(im2, ax=ax)





    def decode_one_var_with_ml(self, target_variable='target_distance', test_size=0.2,
                               models_to_use=['rf', 'nn', 'lr'], cv_folds=5):
        """
        Decode target representation using machine learning approaches.

        Parameters:
        -----------
        target_variable : str or list
            Target variable(s) to predict
        test_size : float
            Proportion of data to use for testing
        models_to_use : list
            List of models to use: 'rf', 'svm', 'nn', 'lr'
        cv_folds : int
            Number of cross-validation folds

        Returns:
        --------
        dict : ML results including model performance and predictions
        """
        print(f"Performing ML-based decoding for target: {target_variable}")

        # Prepare data
        X = self.neural_data.fillna(0).values

        if isinstance(target_variable, str):
            if target_variable not in self.target_data.columns:
                available_cols = [col for col in self.target_data.columns
                                  if target_variable.lower() in col.lower()]
                if available_cols:
                    target_variable = available_cols[0]
                    print(f"Using {target_variable} as target variable")
                else:
                    print(
                        f"Target variable {target_variable} not found. Available columns:")
                    print(self.target_data.columns.tolist())
                    return None

            y = self.target_data[target_variable].fillna(0).values
        else:
            # Multiple target variables
            y = self.target_data[target_variable].fillna(0).values

        # Determine if this is a classification or regression problem
        is_classification = len(
            np.unique(y)) <= 10 and np.all(y == y.astype(int))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if is_classification else None
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers[f'ml_{target_variable}'] = scaler

        # Initialize models
        if is_classification:
            model_dict = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='linear', random_state=42),
                'nn': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
                'lr': LogisticRegression(random_state=42, max_iter=1000)
            }
            scoring = 'accuracy'
        else:
            model_dict = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'svm': SVR(kernel='linear'),
                'nn': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
                'lr': Ridge(random_state=42)
            }
            scoring = 'r2'

        # Train and evaluate models
        ml_results = {}

        for model_name in models_to_use:
            if model_name not in model_dict:
                print(f"Model {model_name} not available. Skipping...")
                continue

            print(f"Training {model_name}...")

            model = model_dict[model_name]

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                        cv=cv_folds, scoring=scoring)

            # Train on full training set
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Evaluation
            if is_classification:
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)

                ml_results[model_name] = {
                    'model': model,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'predictions': y_pred_test,
                    'true_values': y_test
                }
            else:
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mse = mean_squared_error(y_train, y_pred_train)
                test_mse = mean_squared_error(y_test, y_pred_test)

                ml_results[model_name] = {
                    'model': model,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'predictions': y_pred_test,
                    'true_values': y_test
                }

        self.models[f'ml_{target_variable}'] = ml_results
        self.results[f'ml_{target_variable}'] = ml_results

        # Print summary
        self._print_ml_summary(ml_results, is_classification)

        return ml_results


    def _print_ml_summary(self, ml_results, is_classification):
        """Print summary of ML results."""
        print("\n" + "="*50)
        print("ML DECODING RESULTS SUMMARY")
        print("="*50)

        for model_name, results in ml_results.items():
            print(f"\n{model_name.upper()}:")
            print(
                f"  CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")

            if is_classification:
                print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
                print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
            else:
                print(f"  Train R²: {results['train_r2']:.4f}")
                print(f"  Test R²: {results['test_r2']:.4f}")
                print(f"  Test MSE: {results['test_mse']:.4f}")


    def plot_ml_results(self, target_variable, model_name='rf'):
        """Plot ML decoding results."""
        ml_key = f'ml_{target_variable}'
        if ml_key not in self.results or model_name not in self.results[ml_key]:
            print(
                f"No ML results available for {target_variable} with {model_name}")
            return

        results = self.results[ml_key][model_name]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Predicted vs True values
        axes[0].scatter(results['true_values'],
                        results['predictions'], alpha=0.6)
        axes[0].plot([results['true_values'].min(), results['true_values'].max()],
                     [results['true_values'].min(), results['true_values'].max()], 'r--')
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(
            f'Predicted vs True: {target_variable} ({model_name})')

        # Residuals
        residuals = results['true_values'] - results['predictions']
        axes[1].scatter(results['predictions'], residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')

        plt.tight_layout()
        plt.show()

    def save_results(self, save_path):
        """Save all results to a file."""
        results_to_save = {
            'results': self.results,
            'data_info': {
                'neural_data_shape': self.neural_data.shape if self.neural_data is not None else None,
                'behavioral_data_shape': self.behavioral_data.shape if self.behavioral_data is not None else None,
                'target_data_shape': self.target_data.shape if self.target_data is not None else None,
                'raw_data_folder_path': self.raw_data_folder_path,
                'bin_width': self.bin_width,
                'window_width': self.window_width
            }
        }

        with open(save_path, 'wb') as f:
            pickle.dump(results_to_save, f)

        print(f"Results saved to {save_path}")


def create_example_decoder(raw_data_folder_path):
    """
    Create an example target decoder with all methods.

    Parameters:
    -----------
    raw_data_folder_path : str
        Path to the raw monkey data folder

    Returns:
    --------
    TargetDecoder : Configured decoder instance
    """
    # Initialize decoder
    decoder = TargetDecoder(raw_data_folder_path)

    # Prepare data
    decoder.prepare_data(exists_ok=True, use_lags=True)

    # Run CCA decoding
    cca_results = decoder.decode_with_cca(n_components=10)

    # Run ML decoding for common target variables
    target_vars = ['target_distance', 'target_angle']
    for target_var in target_vars:
        try:
            decoder.decode_one_var_with_ml(target_variable=target_var,
                                           models_to_use=['rf', 'svm'],
                                           cv_folds=3)
        except Exception as e:
            print(f"ML decoding failed for {target_var}: {e}")

    return decoder


if __name__ == "__main__":
    # Example usage
    raw_data_path = "all_monkey_data/raw_monkey_data/monkey_Bruno/data_0328"

    # Create and run decoder
    decoder = create_example_decoder(raw_data_path)

    # Plot results
    if 'cca' in decoder.results:
        decoder.plot_cca_results()

    # Save results
    decoder.save_results("target_decoding_results.pkl")
