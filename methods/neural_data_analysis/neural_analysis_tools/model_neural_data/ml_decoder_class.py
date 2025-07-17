import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import matplotlib.pyplot as plt


class MLBehavioralDecoder:
    """
    A class for machine learning-based behavioral variable decoding from neural data.

    This class handles the training and evaluation of various ML models
    for both classification and regression tasks in neural decoding of behavioral variables.
    Can be used to decode any behavioral variable including target properties, movement variables,
    eye position, velocity, etc.
    """

    def __init__(self):
        """Initialize the ML decoder."""
        self.models = {}
        self.scalers = {}
        self.results = {}

    def decode_variable(self, neural_data, behavioral_data, variable='target_distance',
                        test_size=0.2, models_to_use=['rf', 'nn', 'lr'], cv_folds=5):
        """
        Decode behavioral variable using machine learning approaches.

        Parameters:
        -----------
        neural_data : pd.DataFrame
            Neural activity data (features)
        behavioral_data : pd.DataFrame  
            Behavioral data containing the variable to predict
        variable : str or list
            Behavioral variable(s) to predict (e.g., 'target_distance', 'velocity_x', 'eye_position')
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
        print(f"Performing ML-based decoding for variable: {variable}")

        # Prepare data
        X_train_scaled, X_test_scaled, y_train, y_test, is_classification = self._prepare_ml_data(
            neural_data, behavioral_data, variable, test_size)

        if X_train_scaled is None:
            return None

        # Get models for this problem type
        model_dict, scoring = self._get_ml_models(is_classification)

        # Train and evaluate models
        ml_results = self._train_and_evaluate_models(
            X_train_scaled, X_test_scaled, y_train, y_test,
            model_dict, models_to_use, cv_folds, scoring, is_classification)

        # Store results
        self.models[f'ml_{variable}'] = ml_results
        self.results[f'ml_{variable}'] = ml_results

        # Print summary
        self._print_ml_summary(ml_results, is_classification)

        return ml_results

    def _prepare_ml_data(self, neural_data, behavioral_data, variable, test_size):
        """Prepare and preprocess data for ML training."""
        X = neural_data.fillna(0).values

        # Handle variable selection
        if isinstance(variable, str):
            if variable not in behavioral_data.columns:
                available_cols = [col for col in behavioral_data.columns
                                  if variable.lower() in col.lower()]
                if available_cols:
                    variable = available_cols[0]
                    print(f"Using {variable} as behavioral variable")
                else:
                    print(
                        f"Behavioral variable {variable} not found")
                    print(behavioral_data.columns.tolist())
                    return None, None, None, None, None

            y = behavioral_data[variable].fillna(0).values
        else:
            y = behavioral_data[variable].fillna(0).values

        # Determine problem type
        is_classification = len(
            np.unique(y)) <= 10 and np.all(y == y.astype(int))

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if is_classification else None)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers[f'ml_{variable}'] = scaler

        return X_train_scaled, X_test_scaled, y_train, y_test, is_classification

    def _get_ml_models(self, is_classification):
        """Get appropriate models based on problem type."""
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

        return model_dict, scoring

    def _train_and_evaluate_models(self, X_train_scaled, X_test_scaled, y_train, y_test,
                                   model_dict, models_to_use, cv_folds, scoring, is_classification):
        """Train and evaluate all specified models."""
        ml_results = {}

        for model_name in models_to_use:
            if model_name not in model_dict:
                print(f"Model {model_name} not available. Skipping...")
                continue

            print(f"Training and evaluating {model_name} using {cv_folds} folds")
            model = model_dict[model_name]

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                        cv=cv_folds, scoring=scoring)

            # Train and predict
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Store results based on problem type
            ml_results[model_name] = self._evaluate_model_performance(
                model, cv_scores, y_train, y_test, y_pred_train, y_pred_test, is_classification)

        return ml_results

    def _evaluate_model_performance(self, model, cv_scores, y_train, y_test,
                                    y_pred_train, y_pred_test, is_classification):
        """Evaluate model performance and return results dictionary."""
        base_results = {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test,
            'true_values': y_test
        }

        if is_classification:
            base_results.update({
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test)
            })
        else:
            base_results.update({
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test)
            })

        return base_results

    def _print_ml_summary(self, ml_results, is_classification):
        """Print a summary of ML results."""
        print("\n" + "="*50)
        print("ML DECODING SUMMARY")
        print("="*50)

        if is_classification:
            metric_name = "Accuracy"
            test_key = "test_accuracy"
            train_key = "train_accuracy"
        else:
            metric_name = "R²"
            test_key = "test_r2"
            train_key = "train_r2"

        for model_name, results in ml_results.items():
            print(f"\n{model_name.upper()}:")
            print(
                f"  CV {metric_name}: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            print(f"  Train {metric_name}: {results[train_key]:.4f}")
            print(f"  Test {metric_name}: {results[test_key]:.4f}")

        print("\n" + "="*50)

    def get_best_model(self, variable, metric='test_accuracy'):
        """
        Get the best performing model for a given behavioral variable.

        Parameters:
        -----------
        variable : str
            Behavioral variable name
        metric : str
            Metric to use for comparison ('test_accuracy', 'test_r2', etc.)

        Returns:
        --------
        tuple : (model_name, model_results)
        """
        if f'ml_{variable}' not in self.results:
            print(f"No results found for behavioral variable: {variable}")
            return None, None

        results = self.results[f'ml_{variable}']

        best_score = -np.inf
        best_model_name = None
        best_results = None

        for model_name, model_results in results.items():
            if metric in model_results:
                score = model_results[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_results = model_results

        return best_model_name, best_results

    def predict_new_data(self, neural_data, variable, model_name=None):
        """
        Make predictions on new neural data using a trained model.

        Parameters:
        -----------
        neural_data : pd.DataFrame or np.array
            New neural data to make predictions on
        variable : str
            Behavioral variable name (to identify the correct model/scaler)
        model_name : str, optional
            Specific model to use. If None, uses the best performing model.

        Returns:
        --------
        np.array : Predictions
        """
        if f'ml_{variable}' not in self.models:
            print(
                f"No trained models found for behavioral variable: {variable}")
            return None

        # Get the model to use
        if model_name is None:
            model_name, _ = self.get_best_model(variable, 'cv_mean')
            if model_name is None:
                return None

        if model_name not in self.models[f'ml_{variable}']:
            print(
                f"Model {model_name} not found for behavioral variable: {variable}")
            return None

        # Prepare data
        if isinstance(neural_data, pd.DataFrame):
            X = neural_data.fillna(0).values
        else:
            X = neural_data

        # Scale data using the saved scaler
        scaler = self.scalers[f'ml_{variable}']
        X_scaled = scaler.transform(X)

        # Make predictions
        model = self.models[f'ml_{variable}'][model_name]['model']
        predictions = model.predict(X_scaled)

        return predictions

    def plot_ml_results(self, variable, model_name='rf'):
        """Plot ML decoding results."""
        ml_key = f'ml_{variable}'
        if ml_key not in self.results or model_name not in self.results[ml_key]:
            print(
                f"No ML results available for {variable} with {model_name}")
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
            f'Predicted vs True: {variable} ({model_name})')

        # Residuals
        residuals = results['true_values'] - results['predictions']
        axes[1].scatter(results['predictions'], residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')

        plt.tight_layout()
        plt.show()

