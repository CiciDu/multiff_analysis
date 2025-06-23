import sys
from machine_learning.ml_methods import ml_methods_class
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pprint
from scipy import stats
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate
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
        bagging = BaggingClassifier(n_estimators=200, max_features=0.9,
                                    bootstrap_features=True, bootstrap=True, random_state=42)
        boosting = AdaBoostClassifier(n_estimators=500, learning_rate=0.05)
        grad_boosting = GradientBoostingClassifier(learning_rate=0.05, max_depth=7, n_estimators=500, subsample=0.5, max_features='sqrt',
                                                   min_samples_split=7, min_samples_leaf=2)
        rf = RandomForestClassifier(
            n_estimators=40, max_depth=10, random_state=0)
        # voting = VotingClassifier(estimators=[('logreg', logreg), ('svm', svm), ('dt', dt), ('bagging', bagging), ('boosting', boosting), ('rf', rf)],
        #                                        #voting='hard' # based on majority vote
        #                                         voting='soft' # based on sum of probability
        # )
        # mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)

        model_list = [gnb, logreg, dt, bagging, boosting,
                      grad_boosting, rf]  # can add voting, mlp
        model_names = ['gnb', 'logreg', 'dt', 'bagging', 'boosting',
                       'grad_boosting', 'rf']  # can add voting, 'mlp' too

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
        model_comparison_df = pd.DataFrame(
            {'model': model_names, 'accuracy': accuracy_list})
        model_comparison_df.sort_values(
            by='accuracy', ascending=False, inplace=True)
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
    confusion_matrix_df.columns = [
        f'Predicted {num}' for num in range(1, confusion_matrix_df.shape[1]+1)]
    confusion_matrix_df.index = [
        f'Actual {num}' for num in range(1, confusion_matrix_df.shape[0]+1)]
    print(confusion_matrix_df)
    return model, y_pred, model_comparison_df


def use_logistic_regression(x_var_df, y_var_df):

    # To get the confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(
        x_var_df, y_var_df, test_size=0.2, random_state=42)

    model = LogisticRegression()
    results = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{confusion_matrix}")

    # Fit the model on the entire dataset for coefficients and p-values
    model = sm.Logit(y_var_df.values.reshape(-1), x_var_df)
    results = model.fit()
    summary_df = pd.DataFrame(
        {'p_value': results.pvalues, 'Coefficient': results.params})
    summary_df['abs_coeff'] = np.abs(summary_df['Coefficient'])
    summary_df.sort_values(by='abs_coeff', ascending=False, inplace=True)

    return summary_df, conf_matrix


def use_logistic_regression_cv(x_var_df, y_var_df, num_folds=5):

    cv_results = cross_validate(
        LogisticRegression(),
        x_var_df,
        y_var_df,
        cv=num_folds,
        scoring='accuracy',
        return_train_score=True
    )

    test_accuracies = cv_results['test_score']
    train_accuracies = cv_results['train_score']

    # Calculate the average accuracy
    test_avg_accuracy = np.mean(test_accuracies)
    train_avg_accuracy = np.mean(train_accuracies)
    print(
        f"Average Model Accuracy on train set (10-fold CV): {train_avg_accuracy}")
    print(f"Average Model Accuracy (10-fold CV): {test_avg_accuracy}")

    return test_accuracies, train_accuracies


def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    F2 = (1 + beta**2) * precision * recall / \
        (beta**2 * precision + recall + 1e-12)
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


def use_neural_network_on_classification_func(X_train, y_train, X_test, y_test, n_epochs=200, batch_size=100):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # loss metric and optimizer
    model = MultilabelModel(
        n_features=X_train.shape[1], n_classes=y_train.shape[1])
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # prepare model and training parameters
    batches_per_epoch = len(X_train) // batch_size
    print(
        f"Training on {len(X_train)} samples for {n_epochs} epochs with {batches_per_epoch} batches per epoch.")

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
            # Otherwise, the gradient would be a combination of the old gradient, which you have already used to update your model parameters and the newly-computed gradient.
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            # acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            acc = F_score(y_pred, y_batch)
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))

        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        # acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
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
            print(
                f"Epoch {epoch} | Train F2={np.mean(epoch_acc):.4f} | Test F2={acc:.4f} | Test accuracy={accuracy:.4f}")

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
