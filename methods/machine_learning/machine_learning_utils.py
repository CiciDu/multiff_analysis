import sys
from planning_analysis import ml_methods_class
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pprint
from scipy import stats
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import math
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import KFold, train_test_split


def use_ml_model_for_regression(X_train, y_train, X_test, y_test, 
                                              model_names=['linreg', 'svr', 'dt', 'bagging', 'boosting', 'grad_boosting', 'rf'],
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
            cv_scores = cross_val_score(model, x_var, y_var, cv=5, scoring=make_scorer(r2_score))
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
    print(model_comparison_df)
        
    model = model_list[np.argmin(mse_list)]
    print("\n")
    print("The model with the lowest mean squared error is:", model, '.')

    model_name = model_names[np.argmin(mse_list)]

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
        # Get feature importances
        feature_importances = model.feature_importances_
        # Assuming you have the feature names in a list called feature_names
        feature_names = X_train.columns
        # Combine feature names and their importances
        features_and_importances = zip(feature_names, feature_importances)
        # Sort the features by importance
        sorted_features_and_importances = sorted(features_and_importances, key=lambda x: x[1], reverse=True)
        chosen_model_info['sorted_features_and_importances'] = sorted_features_and_importances

    return model_comparison_df, chosen_model_info



def use_ml_model_for_classification(X_train, y_train, X_test, y_test, model=None):
    if model is not None:
        model = model
        model.fit(X_train, y_train)
        model_comparison_df = None
        
    else:
        # try all model options and then choose the one with the highest accuracy
        gnb = GaussianNB()
        logreg = LogisticRegression()
        svm = SVC(probability=True)
        dt = DecisionTreeClassifier()
        bagging = BaggingClassifier(n_estimators=200, max_features=0.9, bootstrap_features=True, bootstrap=True, random_state=42)
        boosting = AdaBoostClassifier(n_estimators=500, learning_rate=0.05)
        grad_boosting = GradientBoostingClassifier(learning_rate=0.05, max_depth=7, n_estimators=500, subsample=0.5, max_features='sqrt', \
                                                    min_samples_split=7, min_samples_leaf=2)
        rf = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=0)
        # voting = VotingClassifier(estimators=[('logreg', logreg), ('svm', svm), ('dt', dt), ('bagging', bagging), ('boosting', boosting), ('rf', rf)], 
        #                                        #voting='hard' # based on majority vote
        #                                         voting='soft' # based on sum of probability
        # )
        #mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)

        model_list = [gnb, logreg, dt, bagging, boosting, grad_boosting, rf] # can add voting, mlp
        model_names = ['gnb', 'logreg', 'dt', 'bagging', 'boosting', 'grad_boosting', 'rf'] # can add voting, 'mlp' too


        # find the model with the highest accuracy
        if len(X_train) < 10000:
            model_list.append(svm)
            model_names.append('svm')
        accuracy_list = []
        for i in range(len(model_list)):
            model = model_list[i]
            model_name = model_names[i]
            print("model:", model_name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_list.append(accuracy)
            # print("model:", model)
            # print("accuracy:", accuracy)
            # print("confusion matrix:", confusion_matrix(y_test, y_pred))

        # make a table to compare the results of all the models in model_list
        model_comparison_df = pd.DataFrame({'model': model_names, 'accuracy': accuracy_list})
        model_comparison_df.sort_values(by='accuracy', ascending=False, inplace=True)
        print(model_comparison_df)
            
        model = model_list[np.argmax(accuracy_list)]
        print("\n")
        print("The model with the highest accuracy is:", model, '!!')


    # predict
    y_pred = model.predict(X_test)
    # evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("chosen model accuracy:", accuracy)
    # confusion matrix
    print("chosen model confusion matrix:", confusion_matrix(y_test, y_pred))
    # make the confusion matrix into a table
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    confusion_matrix_df.columns = [f'Predicted {num}' for num in range(1, confusion_matrix_df.shape[1]+1)]
    confusion_matrix_df.index = [f'Actual {num}' for num in range(1, confusion_matrix_df.shape[0]+1)]
    print(confusion_matrix_df)
    return model, y_pred, model_comparison_df




# Source for below: https://www.kaggle.com/code/juanmah/grid-search-utils
# Note: functions were slightly modified

"""Utility script with functions to be used with the results of GridSearchCV.

**plot_grid_search** plots as many graphs as parameters are in the grid search results.

**table_grid_search** shows tables with the grid search results.

Inspired in [Displaying the results of a Grid Search](https://www.kaggle.com/grfiv4/displaying-the-results-of-a-grid-search) notebook,
of [George Fisher](https://www.kaggle.com/grfiv4)
"""



__author__ = "Juanma Hernández"
__copyright__ = "Copyright 2019"
__credits__ = ["Juanma Hernández", "George Fisher"]
__license__ = "GPL"
__maintainer__ = "Juanma Hernández"
__email__ = "https://twitter.com/juanmah"
__status__ = "Utility script"


def plot_grid_search(clf):
    """Plot as many graphs as parameters are in the grid search results.

    Each graph has the values of each parameter in the X axis and the Score in the Y axis.

    Parameters
    ----------
    clf: estimator object result of a GridSearchCV
        This object contains all the information of the cross validated results for all the parameters combinations.
    """
    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])

    # Get parameters
    parameters=cv_results['params'][0].keys()

    # Calculate the number of rows and columns necessary
    rows = -(-len(parameters) // 2)
    columns = min(len(parameters), 2)
    # Create the subplot
    fig = make_subplots(rows=rows, cols=columns)
    # Initialize row and column indexes
    row = 1
    column = 1

    # For each of the parameters
    for parameter in parameters:
        # As all the graphs have the same traces, and by default all traces are shown in the legend,
        # the description appears multiple times. Then, only show legend of the first graph.
        if row == 1 and column == 1:
            show_legend = True
        else:
            show_legend = False

        # Mean test score
        mean_test_score = cv_results[cv_results['rank_test_score'] != 1]
        fig.add_trace(go.Scatter(
            name='Mean test score',
            x=mean_test_score['param_' + parameter],
            y=mean_test_score['mean_test_score'],
            mode='markers',
            marker=dict(size=mean_test_score['mean_fit_time'],
                        color='SteelBlue',
                        sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                        sizemin=4,
                        sizemode='area'),
            text=mean_test_score['params'].apply(
                lambda x: pprint.pformat(x, width=-1).replace('{', '').replace('}', '').replace('\n', '<br />')),
            showlegend=show_legend), 
            row=row,
            col=column)

        # Best estimators
        rank_1 = cv_results[cv_results['rank_test_score'] == 1]
        fig.add_trace(go.Scatter(
            name='Best estimators',
            x=rank_1['param_' + parameter],
            y=rank_1['mean_test_score'],
            mode='markers',
            marker=dict(size=rank_1['mean_fit_time'],
                        color='Crimson',
                        sizeref=2. * cv_results['mean_fit_time'].max() / (40. ** 2),
                        sizemin=4,
                        sizemode='area'),
            text=rank_1['params'].apply(str),
            showlegend=show_legend),
            row=row,
            col=column)

        fig.update_xaxes(title_text=parameter, row=row, col=column)
        fig.update_yaxes(title_text='Score', row=row, col=column)

        # Check the linearity of the series
        # Only for numeric series
        if pd.to_numeric(cv_results['param_' + parameter], errors='coerce').notnull().all():
            x_values = cv_results['param_' + parameter].sort_values().unique().tolist()
            r = stats.linregress(x_values, range(0, len(x_values))).rvalue
            # If not so linear, then represent the data as logarithmic
            if r < 0.86:
                fig.update_xaxes(type='log', row=row, col=column)

        # Increment the row and column indexes
        column += 1
        if column > columns:
            column = 1
            row += 1

            # Show first the best estimators
    fig.update_layout(legend=dict(traceorder='reversed'),
                      width=columns * 560 + 100,
                      height=rows * 460,
                    #   title=dict(text='Best score: {:.3f} with {}'.format(cv_results['mean_test_score'].iloc[0],
                    #                                             str(cv_results['params'].iloc[0]).replace('{',
                    #                                                                                       '').replace(
                    #                                                 '}', '')),
                    #              font=dict(size=14), automargin=True, yref='paper'),
                      hovermode='closest',
                      template='none')
    fig.show()




def table_grid_search(clf, all_columns=False, all_ranks=False, save=True):
    """Show tables with the grid search results.

    Parameters
    ----------
    clf: estimator object result of a GridSearchCV
        This object contains all the information of the cross validated results for all the parameters combinations.

    all_columns: boolean, default: False
        If true all columns are returned. If false, the following columns are dropped:

        - params. As each parameter has a column with the value.
        - std_*. Standard deviations.
        - split*. Split scores.

    all_ranks: boolean, default: False
        If true all ranks are returned. If false, only the rows with rank equal smaller than 6 are returned.

    save: boolean, default: True
        If true, results are saved to a CSV file.
    """
    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])

    # Reorder
    columns = cv_results.columns.tolist()
    # rank_test_score first, mean_test_score second and std_test_score third
    columns = columns[-1:] + columns[-3:-1] + columns[:-3]
    cv_results = cv_results[columns]

    if save:
        cv_results.to_csv('--'.join(cv_results['params'][0].keys()) + '.csv', index=True, index_label='Id')

    # Unless all_columns are True, drop not wanted columns: params, std_* split*
    if not all_columns:
        cv_results.drop('params', axis='columns', inplace=True)
        cv_results.drop(list(cv_results.filter(regex='^std_.*')), axis='columns', inplace=True)
        cv_results.drop(list(cv_results.filter(regex='^split.*')), axis='columns', inplace=True)

    # Unless all_ranks are True, only keep the top 20 results
    if not all_ranks:
        cv_results = cv_results[cv_results['rank_test_score'] <= 20]

    return cv_results


def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)



class MultilabelModel(nn.Module):
    def __init__(self, n_features=4, n_classes=3):
        super().__init__()
        self.hidden = nn.Linear(n_features, 200)
        self.act = nn.ReLU()
        self.output = nn.Linear(200, n_classes)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x
    

def use_neural_network_on_classification_func(X_train, y_train, X_test, y_test, n_epochs = 200, batch_size = 100):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    

    # loss metric and optimizer
    model = MultilabelModel(n_features=X_train.shape[1], n_classes=y_train.shape[1])
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # prepare model and training parameters
    batches_per_epoch = len(X_train) // batch_size
    print(f"Training on {len(X_train)} samples for {n_epochs} epochs with {batches_per_epoch} batches per epoch.")
    
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    
    # training loop
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        # set model in training mode and run through each batch
        model.train()
        for i in range(batches_per_epoch):
            # take a batch
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad() # Otherwise, the gradient would be a combination of the old gradient, which you have already used to update your model parameters and the newly-computed gradient.
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            ## acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            acc = F_score(y_pred, y_batch)
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))

        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        #acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
        acc = F_score(y_pred, y_test)
        ce = float(ce)
        acc = float(acc)
        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            y_test_np = y_test.detach().numpy()
            y_pred_np = y_pred.detach().numpy().round().astype(int)
            accuracy = accuracy_score(y_test_np, y_pred_np)
            print(f"Epoch {epoch} | Train F2={np.mean(epoch_acc):.4f} | Test F2={acc:.4f} | Test accuracy={accuracy:.4f}")
    
    # Restore best model
    model.load_state_dict(best_weights)
    
    # Plot the loss and accuracy
    plt.plot(train_loss_hist, label="train")
    plt.plot(test_loss_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()
    
    
    plt.plot(train_acc_hist, label="train")
    plt.plot(test_acc_hist, label="test")
    plt.xlabel("epochs")
    plt.ylabel("F score")
    plt.legend()
    plt.show()
    
    return model, y_pred_np


import torch.nn as nn


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
    

# # Define the neural network model
# class MultiLayerRegression(nn.Module):
#     def __init__(self, input_size):
#         super(MultiLayerRegression, self).__init__()
#         self.layer1 = nn.Linear(input_size, 64)  # First layer
#         self.relu = nn.ReLU()                    # Activation function
#         self.layer2 = nn.Linear(64, 32)          # Hidden layer
#         self.output_layer = nn.Linear(32, 1)     # Output layer for regression

#     def forward(self, x):
#         x = self.relu(self.layer1(x))
#         x = self.relu(self.layer2(x))
#         x = self.output_layer(x)
#         return x

# Example usage
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
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.show()

    return model, predictions


def use_linear_regression(X_train, X_test, y_train, y_test,
                          show_plot=True):
        
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)


    model = sm.OLS(y_train, X_train)
    results = model.fit()

    summary_df = pd.DataFrame({'p_value': results.pvalues, 'Coefficient': results.params, 'Std Err': results.bse, 't': results.tvalues})

    # print(results.summary())
    print("R-squared: ", round(results.rsquared, 4))
    print("Adjusted R-squared: ", round(results.rsquared_adj, 4))

    y_pred = results.predict(X_test)

    r_squared_on_test = r2_score(y_test, y_pred)
    print("R-squared on test set:", round(r_squared_on_test, 4))

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Create a scatter plot
    if show_plot:
        plt.scatter(y_test, y_pred)
        # draw a line of y = x
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        plt.xlabel('Real Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Real vs Predicted Values with RMSE on test set: {math.sqrt(mse)}')
        plt.show()

    # If you want the summary statistics of the coefficients, you can do:
    summary_df['abs_coeff'] = np.abs(summary_df['Coefficient'])
    summary_df.sort_values(by='abs_coeff', ascending=False, inplace=True)
    return summary_df, y_pred, results, r_squared_on_test


def use_linear_regression_cv(x_var, y_var):
    # also try cross validation
    model = LinearRegression()
    
    # Perform cross-validation
    # cv=5 specifies the number of folds in K-Fold cross-validation
    # You can adjust the scoring parameter based on your requirements
    cv_scores = cross_val_score(model, x_var, y_var, cv=10, scoring=make_scorer(r2_score))
    
    # Calculate the average R-squared across all folds
    avg_r_squared = cv_scores.mean()
    
    # You can also calculate other statistics like standard deviation to assess variability
    std_r_squared = cv_scores.std()

    print("avg_r_squared from cv:", round(avg_r_squared, 4))

    return avg_r_squared, std_r_squared



def use_logistic_regression(x_var_df, y_var_df):

    # Assuming mtc.x_var_df is your DataFrame with features and 'target' is the target variable
    X = x_var_df.copy()
    y = y_var_df.copy()  # Replace 'target' with the actual target column name

    # Initialize KFold with 10 splits
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    train_accuracies = []
    accuracies = []

    # Use the Lasso model for feature selection
    lasso = Lasso(alpha=0.01)  # Adjust alpha as needed
    lasso.fit(x_var_df.values, y_var_df.values.reshape(-1))
    selected_features = X.columns[(lasso.coef_ != 0)]
    X_selected = X[selected_features]
    num_selected_features = X_selected.shape[1]


    # To get the confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    # Fit the model
    results = model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")


    # Fit the model on the entire dataset for coefficients and p-values
    model = sm.Logit(y_var_df.values.reshape(-1), X_selected)
    results = model.fit()
    summary_df = pd.DataFrame({'p_value': results.pvalues, 'Coefficient': results.params})
    summary_df['abs_coeff'] = np.abs(summary_df['Coefficient'])
    summary_df.sort_values(by='abs_coeff', ascending=False, inplace=True)


    for train_index, test_index in kf.split(X_selected):
        
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize the Logistic Regression model
        model = LogisticRegression()

        # Fit the model
        model.fit(X_train, y_train.values.reshape(-1))

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Also calculate accuracy on train set
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        train_accuracies.append(train_accuracy)


    # Calculate the average accuracy
    test_avg_accuracy = np.mean(accuracies)
    train_avg_accuracy = np.mean(train_accuracies)
    print(f"Average Model Accuracy on train set (10-fold CV): {train_avg_accuracy}")
    print(f"Average Model Accuracy (10-fold CV): {test_avg_accuracy}")

    return summary_df, test_avg_accuracy, train_avg_accuracy, num_selected_features

    # # ===========
    # # Initialize KFold with 10 splits
    # kf = KFold(n_splits=10, shuffle=True, random_state=1)
    
    # accuracies = []
    
    # for train_index, test_index in kf.split(x_var_df):
    #     X_train, X_test = x_var_df.iloc[train_index], x_var_df.iloc[test_index]
    #     y_train, y_test = y_var_df.iloc[train_index], y_var_df.iloc[test_index]
        
    #     # Fit the model
    #     model = sm.Logit(y_train, X_train)
    #     results = model.fit(disp=0)
        
    #     # Predict
    #     predictions = results.predict(X_test)
    #     # Convert probabilities to binary outcome
    #     predictions_binary = np.where(predictions > 0.5, 1, 0)
        
    #     # Calculate accuracy
    #     accuracy = accuracy_score(y_test, predictions_binary)
    #     accuracies.append(accuracy)
    
    # # Calculate the average accuracy
    # average_accuracy = np.mean(accuracies)
    # print(f"Average Model Accuracy (10-fold CV): {average_accuracy}")
    
    # # Fit the model on the entire dataset for coefficients and p-values
    # model = sm.Logit(y_var_df, x_var_df)
    # results = model.fit()
    
    # summary_df = pd.DataFrame({'p_value': results.pvalues, 'Coefficient': results.params})
    # summary_df['abs_coeff'] = np.abs(summary_df['Coefficient'])
    # summary_df.sort_values(by='abs_coeff', ascending=False, inplace=True)
    #return summary_df, results, average_accuracy


def use_classification(mtc, y_var_column):
    ml_inst = ml_methods_class.MlMethods()
    ml_inst.data_source = mtc

    all_info = pd.DataFrame()
    for test_or_control in ['test', 'control']:
        for ref_columns_only in [True, False]:
            print(' ')
            print('=======================================================')
            mtc.test_or_control = test_or_control
            ml_inst.data_source.streamline_preparing_for_ml(y_var_column, 
                                                ref_columns_only=ref_columns_only,
                                                cluster_to_keep='all',
                                                cluster_for_interaction='stop_ff_cluster_100',
                                                add_ref_interaction=True,
                                                winsorize_angle_features=True,
                                                using_lasso=False, 
                                                use_combd_features_for_cluster_only=False,
                                                for_classification=True
                                                )
            if (ml_inst.data_source.x_var_df.shape[0] == 0) or (ml_inst.data_source.x_var_df.shape[1] == 0):
                print('no data for y_var_column:', y_var_column)
                continue

            ml_inst.use_logistic_regression(mtc.x_var_df, mtc.y_var_df)

            print('num_features:', ml_inst.data_source.x_var_df.shape[1])
            print('num_selected_features:', ml_inst.num_selected_features)
            print('sample_size:', ml_inst.data_source.x_var_df.shape[0])
            temp_info = {'average_accuracy': ml_inst.average_accuracy,
                         'train_avg_accuracy': ml_inst.train_avg_accuracy,
                        'y_var_column': y_var_column,
                        'test_or_control': test_or_control,
                        'ref_columns_only': ref_columns_only,
                        'sample_size': ml_inst.data_source.x_var_df.shape[0],
                        'num_features': ml_inst.data_source.x_var_df.shape[1],
                        'num_selected_features': ml_inst.num_selected_features,
                        }
            temp_info = pd.DataFrame(temp_info, index=[0])

            all_info = pd.concat([all_info, temp_info], axis=0)
    return all_info, ml_inst