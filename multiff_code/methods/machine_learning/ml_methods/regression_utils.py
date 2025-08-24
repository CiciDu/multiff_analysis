from machine_learning.ml_methods import ml_methods_utils
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import cross_validate

def use_linear_regression(X_train, X_test, y_train, y_test,
                          show_plot=True, y_var_name=None):
    # Ensure y is 1-dimensional
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Add constant to X for intercept
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Fit model
    model = sm.OLS(y_train, X_train_const)
    results = model.fit()

    # Predict
    y_pred = results.predict(X_test_const)

    # Metrics
    r2_train = results.rsquared
    r2_adj = results.rsquared_adj
    r2_test = round(1 - sum((y_test - y_pred) ** 2) /
                    sum((y_test - np.mean(y_test)) ** 2), 4)
    pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Detect regression type
    reg_type = "Simple" if X_train.shape[1] == 1 else "Multiple"

    # Print metrics
    if y_var_name is not None:
        print(f"\n--- {reg_type} Linear Regression: {y_var_name} ---")
    else:
        print(f"\n--- {reg_type} Linear Regression ---")
    print(f"R-squared (train):        {r2_train:.4f}")
    print(f"Adjusted R-squared:       {r2_adj:.4f}")
    print(f"R-squared (test):         {r2_test:.4f}")
    print(f"Pearson Corr (test):      {pearson_corr:.4f}")
    print(f"MAE (test):               {mae:.4f}")
    print(f"MSE (test):               {mse:.4f}")
    print(f"RMSE (test):              {rmse:.4f}")

    # Plot
    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        title = f"{y_var_name + ' — ' if y_var_name else ''}{reg_type} Linear Regression\n"
        title += f"Test $R^2$: {r2_test:.3f}   |   Pearson $R$: {pearson_corr:.3f}"
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Coefficient summary
    summary_df = pd.DataFrame({
        'Coefficient': results.params,
        'Std Err': results.bse,
        't': results.tvalues,
        'p_value': results.pvalues
    })

    summary_df = ml_methods_utils.process_summary_df(summary_df)

    return summary_df, y_pred, results, r2_test


def ml_model_for_regression(X_train, y_train, X_test, y_test,
                                model_names=[
                                    'linreg', 'svr', 'dt', 'bagging', 'boosting', 'grad_boosting', 'rf'],
                                use_cv=False):

    models = {'linreg': LinearRegression(),
              'svr': SVR(),
              'dt': DecisionTreeRegressor(),
              'bagging': BaggingRegressor(n_estimators=100, max_samples=0.5, bootstrap_features=True, bootstrap=True, random_state=42),
              'boosting': AdaBoostRegressor(n_estimators=100, learning_rate=0.05),
              'grad_boosting': GradientBoostingRegressor(min_samples_split=50,
                                                         min_samples_leaf=10,
                                                         max_depth=5,
                                                         max_features=0.3,
                                                         n_iter_no_change=10,
                                                         ),
              'rf': RandomForestRegressor(random_state=42,
                                          min_samples_split=50,
                                          min_samples_leaf=10,
                                          max_features=0.3,
                                          n_jobs=-1,
                                          ),
              }
    
    if model_names is None:
        model_names = list(models.keys())

    model_comparison_df, model_list, mse_list = _get_model_comparison_df(
        X_train, y_train, X_test, y_test, model_names, models, use_cv)

    # find the model with the lowest mean squared error
    model = model_list[np.argmin(mse_list)]
    model_name = model_names[np.argmin(mse_list)]
    # compile into chosen_model_info
    chosen_model_info = _choose_best_model(
        model, model_name, X_train, X_test, y_test)

    return model_comparison_df, chosen_model_info


def _get_model_comparison_df(X_train, y_train, X_test, y_test, model_names, models, use_cv=False):
    # find the model with the lowest mean squared error
    model_list = []
    mse_list = []
    r_squared_list = []
    avg_r_squared_list = []
    std_r_squared_list = []

    for model_name in model_names:
        model = models[model_name]
        model_list.append(model)
        print("model:", model_name)

        if use_cv:
            print('Running Cross Validation...')
            x_var = pd.concat([X_train, X_test])
            y_var = pd.concat([y_train, y_test])
            cv_scores = cross_val_score(
                model, x_var, y_var, cv=5, scoring=make_scorer(r2_score))
            # Calculate the average R-squared across all folds
            avg_r_squared_list.append(cv_scores.mean())
            std_r_squared_list.append(cv_scores.std())

        # fit the model in the normal way
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_list.append(mean_squared_error(y_test, y_pred))
        r_squared_list.append(r2_score(y_test, y_pred))

    # make a table to compare the results of all the models in model_list
    model_comparison_df = pd.DataFrame({'model': model_names,
                                        'mse': mse_list,
                                        'r_squared_test': r_squared_list})
    if len(avg_r_squared_list) > 0:
        model_comparison_df['avg_r_squared'] = avg_r_squared_list
        model_comparison_df['std_r_squared'] = std_r_squared_list

    model_comparison_df.sort_values(by='mse', ascending=True, inplace=True)
    return model_comparison_df, model_list, mse_list


def _choose_best_model(model, model_name, X_train, X_test, y_test):

    print("\n")
    print("The model with the lowest mean squared error is:", model, '.')
    # predict
    y_pred = model.predict(X_test)
    # evaluate
    mse = mean_squared_error(y_test, y_pred)
    print("chosen model mse:", mse)

    chosen_model_info = {'model': model,
                         'y_pred': y_pred,
                         'mse': mse,
                         'r_squared_test': r2_score(y_test, y_pred),
                         }

    if model_name == 'rf':
        chosen_model_info['sorted_features_and_importances'] = _get_rf_feature_importances(
            model, X_train, feature_names=X_train.columns)

    return chosen_model_info


def _get_rf_feature_importances(model, feature_names=None):
    feature_importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"Neuron_{i}" for i in range(
            len(feature_importances))]
    # Combine feature names and their importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    return importance_df


def plot_feature_importance(importance_df, predictor_var):
    plt.figure(figsize=(10, 6))
    plt.bar(range(min(15, len(importance_df))),
            importance_df['importance'].head(15))
    plt.title(f'Feature Importance: {predictor_var}')
    plt.xlabel('Neuron Index (sorted by importance)')
    plt.ylabel('Importance')
    plt.xticks(range(min(15, len(importance_df))),
               importance_df['feature'].head(15), rotation=45)
    plt.tight_layout()
    plt.show()


def pearson_r_score(y_true, y_pred):
    # Pearson r as a scorer, ignoring p-value
    return pearsonr(y_true, y_pred)[0]

def use_linear_regression_cv(x_var, y_var, cv=10, groups=None, verbose=False):
    """
    Perform cross-validation with linear regression.
    
    Parameters:
    - x_var: feature matrix
    - y_var: target vector
    - cv: int or cross-validation splitter
    - groups: optional group labels (for GroupKFold)
    
    Returns:
    - dict of train/test metrics (mean ± std) including R² and Pearson r.
    """
    scoring = {
        'r2': make_scorer(r2_score),
        'pearson_r': make_scorer(pearson_r_score)
    }
    
    cv_results = cross_validate(
        LinearRegression(),
        x_var,
        y_var,
        cv=cv,
        scoring=scoring,
        groups=groups,
        return_train_score=True
    )

    metrics = ['r2', 'pearson_r']
    results = {}

    if verbose:
        print(f"{'Metric':<12} {'Train Mean':>12} {'Test Mean':>12} {'Train Std':>12}   {'Test Std':>12}")
        print("-" * 70)

    # Then print metrics
    for metric in metrics:
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        train_mean, train_std = np.mean(train_scores), np.std(train_scores)
        test_mean, test_std = np.mean(test_scores), np.std(test_scores)

        results[f'train_{metric}'] = (train_mean, train_std)
        results[f'test_{metric}'] = (test_mean, test_std)

        if verbose:
            print(f"{metric.capitalize():<12} {train_mean:12.4f} {test_mean:12.4f} {train_std:12.4f}   {test_std:12.4f}")

    return results


# def use_linear_regression_cv(x_var, y_var, num_folds=10):
#     kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
#     r2_train_scores = []
#     r2_test_scores = []
#     pearson_test_scores = []
#     pearson_p_values = []

#     for train_index, test_index in kf.split(x_var):
#         x_train, x_test = x_var[train_index], x_var[test_index]
#         y_train, y_test = y_var[train_index], y_var[test_index]

#         model = LinearRegression()
#         model.fit(x_train, y_train)

#         y_train_pred = model.predict(x_train)
#         y_test_pred = model.predict(x_test)

#         # R² scores
#         r2_train_scores.append(r2_score(y_train, y_train_pred))
#         r2_test_scores.append(r2_score(y_test, y_test_pred))

#         # Pearson r and p-value on test set
#         r, p = pearsonr(y_test, y_test_pred)
#         pearson_test_scores.append(r)
#         pearson_p_values.append(p)

#     # Summary stats
#     train_r2_mean = np.mean(r2_train_scores)
#     train_r2_std = np.std(r2_train_scores)
#     test_r2_mean = np.mean(r2_test_scores)
#     test_r2_std = np.std(r2_test_scores)
#     pearson_r_mean = np.mean(pearson_test_scores)
#     pearson_r_std = np.std(pearson_test_scores)
#     pearson_p_mean = np.mean(pearson_p_values)
#     pearson_p_std = np.std(pearson_p_values)

#     print(f"{'Metric':<35}{'Mean':>10} ± {'Std':<10}")
#     print("-" * 60)
#     print(f"{'Train R²':<35}{train_r2_mean:>10.4f} ± {train_r2_std:<10.4f}")
#     print(f"{'Test R²':<35}{test_r2_mean:>10.4f} ± {test_r2_std:<10.4f}")
#     print(f"{'Test Pearson r':<35}{pearson_r_mean:>10.4f} ± {pearson_r_std:<10.4f}")
#     print(f"{'Test Pearson p-value':<35}{pearson_p_mean:>10.4g} ± {pearson_p_std:<10.4g}")

#     return {
#         "train_r2_mean": train_r2_mean,
#         "train_r2_std": train_r2_std,
#         "test_r2_mean": test_r2_mean,
#         "test_r2_std": test_r2_std,
#         "pearson_r_mean": pearson_r_mean,
#         "pearson_r_std": pearson_r_std,
#         "pearson_p_mean": pearson_p_mean,
#         "pearson_p_std": pearson_p_std
#     }




# def use_linear_regression_cv(x_var, y_var, num_folds=10):
#     # also try cross validation
#     model = LinearRegression()

#     # Perform cross-validation
#     # cv specifies the number of folds in K-Fold cross-validation
#     # You can adjust the scoring parameter based on your requirements
#     cv_results = cross_validate(
#         model,
#         x_var,
#         y_var,
#         cv=num_folds,
#         scoring='r2',
#         return_train_score=True
#     )

#     test_scores = cv_results['test_score']
#     train_scores = cv_results['train_score']

#     # Calculate the average R-squared across all folds
#     test_avg_r_squared = test_scores.mean()
#     train_avg_r_squared = train_scores.mean()

#     # You can also calculate other statistics like standard deviation to assess variability
#     test_std_r_squared = test_scores.std()
#     train_std_r_squared = train_scores.std()

#     print(
#         f"Average R-squared on train set ({num_folds}-fold CV): {round(train_avg_r_squared, 4)}")
#     print(
#         f"Average R-squared on test set ({num_folds}-fold CV): {round(test_avg_r_squared, 4)}")

#     return test_avg_r_squared, test_std_r_squared, train_avg_r_squared, train_std_r_squared


class MultiLayerRegression(nn.Module):
    def __init__(self, input_size, hidden_layers=[128]):
        super(MultiLayerRegression, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.relu = nn.ReLU()

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))

        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x


def use_neural_network_on_linear_regression_func(X_train, y_train, X_test, y_test, learning_rate=0.0005, epochs=200):
    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Initialize the model
    input_size = X_train.shape[1]
    model = MultiLayerRegression(input_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

    # plot test results
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    # also plot a line of y=x
    plt.plot([y_test.min(), y_test.max()], [
             y_test.min(), y_test.max()], 'k--', lw=4)
    plt.show()

    return model, predictions


def get_significant_features_in_one_row(summary_df, max_features_to_save=None, add_coeff=True):
    summary_df = summary_df.copy()
    summary_df = summary_df.reset_index(drop=False).copy()
    summary_df.rename(columns={'index': 'feature'}, inplace=True)
    if max_features_to_save is not None:
        summary_df = summary_df.set_index(
            'rank_by_abs_coeff').iloc[:max_features_to_save].copy()

    summary_df.index = summary_df.index.astype(str)
    temp_info = summary_df[['feature']].T.reset_index(drop=True).copy()
    if add_coeff:
        temp_info2 = summary_df[['Coefficient']].copy()
        temp_info2.index = 'coeff_' + np.array(summary_df.index.astype(str))
        temp_info2 = temp_info2.T.reset_index(drop=True)
        temp_info = pd.concat([temp_info, temp_info2], axis=1)

    if temp_info.shape[0] > 1:
        raise ValueError('temp_info should only have one row')
    temp_info.columns.name = ''
    return temp_info


# def get_significant_features_in_one_row(summary_df, max_features_to_save=None, add_coeff=True):
#     summary_df = summary_df.copy()
#     summary_df.rename(columns={'index': 'feature'}, inplace=True)
#     if max_features_to_save is not None:
#         summary_df = summary_df.set_index(
#             'rank_by_abs_coeff').iloc[:max_features_to_save].copy()
#     summary_df.index = summary_df.index.astype(str)
#     temp_info = summary_df[['feature']].T.reset_index(drop=True).copy()
#     if add_coeff:
#         temp_info2 = summary_df[['Coefficient']].copy()
#         temp_info2.index = 'coeff_' + np.array(summary_df.index.astype(str))
#         temp_info2 = temp_info2.T.reset_index(drop=True)
#         temp_info = pd.concat([temp_info, temp_info2], axis=1)

#     if temp_info.shape[0] > 1:
#         raise ValueError('temp_info should only have one row')
#     temp_info.columns.name = ''
#     return temp_info






