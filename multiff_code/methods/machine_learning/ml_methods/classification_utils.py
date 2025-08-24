from machine_learning.ml_methods import ml_methods_utils

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, classification_report,
    confusion_matrix
)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# Optional external libs (guard if unavailable):
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


def _proba_or_score(mdl, X):
    """Return class probabilities if available, else decision scores (for ROC AUC)."""
    if hasattr(mdl, "predict_proba"):
        return mdl.predict_proba(X)
    if hasattr(mdl, "decision_function"):
        return mdl.decision_function(X)
    return None  # ROC AUC will be skipped


def ml_model_for_classification(X_train, y_train, X_test, y_test, model_names=None, kfold_cv=None, random_state=42, verbose=True):
    """
    Train and evaluate multiple classifiers.
      - If kfold_cv is None: fit on train, evaluate on test.
      - If kfold_cv is int or CV splitter: report cross-validated accuracy on train.
    Returns: (best_model, y_pred_best, model_comparison_df)
    """

    # ---- Define candidate models (list of (name, estimator)) ----
    all_models = [
        ("gnb", GaussianNB()),
        ("logreg", LogisticRegression(max_iter=200, n_jobs=None)
         if "n_jobs" in LogisticRegression().get_params() else LogisticRegression(max_iter=200)),
        ("ridge", RidgeClassifier()),
        ("sgd", SGDClassifier(max_iter=1000, tol=1e-3, random_state=random_state)),
        ("dt", DecisionTreeClassifier(random_state=random_state)),
        ("bagging", BaggingClassifier(
            n_estimators=200, max_features=0.9, bootstrap_features=True, bootstrap=True, random_state=random_state
        )),
        ("rf", RandomForestClassifier(n_estimators=40,
         max_depth=10, random_state=random_state)),
        ("extra_trees", ExtraTreesClassifier(
            n_estimators=200, random_state=random_state)),
        ("boosting", AdaBoostClassifier(n_estimators=500,
         learning_rate=0.05, random_state=random_state)),
        ("grad_boosting", GradientBoostingClassifier(
            learning_rate=0.05, max_depth=7, n_estimators=500, subsample=0.5,
            max_features="sqrt", min_samples_split=7, min_samples_leaf=2, random_state=random_state
        )),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100,),
         max_iter=500, random_state=random_state)),
    ]

    # Add optional libraries if available
    if XGBClassifier is not None:
        all_models.append(("xgb", XGBClassifier(
            random_state=random_state, eval_metric="mlogloss", use_label_encoder=False
        )))
    if LGBMClassifier is not None:
        all_models.append(("lgbm", LGBMClassifier(random_state=random_state)))
    if CatBoostClassifier is not None:
        all_models.append(("catboost", CatBoostClassifier(
            verbose=0, random_state=random_state)))
    # Add SVM only for smaller datasets (probability=True is slower)
    if len(X_train) < 10_000:
        all_models.append(
            ("svm", SVC(probability=True, random_state=random_state)))

    # Subselect by model_names if given
    if model_names is not None:
        name_set = set(model_names)
        all_models = [(n, m) for n, m in all_models if n in name_set]

    metrics_list = []

    # CV setup (use stratified for classification)
    cv = None
    if kfold_cv:
        cv = StratifiedKFold(n_splits=kfold_cv, shuffle=True, random_state=random_state) if isinstance(
            kfold_cv, int) else kfold_cv

    # ---- Evaluate models ----
    for name, mdl in all_models:
        if verbose:
            print(f"Model: {name}")

        if cv is not None:
            acc_scores = cross_val_score(
                mdl, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
            metrics_list.append({
                "model": name,
                "accuracy": acc_scores.mean(),
                "balanced_accuracy": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "mcc": None,
                "roc_auc": None,
                "cv_scores": acc_scores,
            })
            if verbose:
                print(
                    f"CV mean accuracy: {acc_scores.mean():.4f}  | scores: {np.round(acc_scores, 4)}")
        else:
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)

            # ROC AUC (binary or multiclass). Try proba, else decision_function.
            roc_auc = None
            try:
                scores = _proba_or_score(mdl, X_test)
                if scores is not None:
                    if scores.ndim == 1:
                        # binary decision_function can be 1-D; need probability-like direction
                        roc_auc = roc_auc_score(y_test, scores)
                    else:
                        # probabilities or OvR scores for multiclass
                        roc_auc = roc_auc_score(
                            y_test, scores, multi_class="ovr")
            except Exception:
                roc_auc = None

            metrics_list.append({
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "roc_auc": roc_auc,
                "cv_scores": None,
            })

    model_comparison_df = pd.DataFrame(metrics_list)
    # If we only did CV, rank by accuracy; if test split, same thing.
    model_comparison_df = model_comparison_df.sort_values(
        by="accuracy", ascending=False).reset_index(drop=True)
    if verbose:
        print("\nModel comparison:\n", model_comparison_df)

    # ---- Pick & fit best model on train, then evaluate on test ----
    best_name = model_comparison_df.loc[0, "model"]
    best_model = dict(all_models)[best_name]  # safe because names are unique
    if verbose:
        print(f"\nBest model by accuracy: {best_name}")

    # If CV mode, we still need a final fit on the full train and a test evaluation:
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    if verbose:
        print("\nChosen model accuracy:", accuracy_score(y_test, y_pred_best))
        print("\nClassification Report:\n", classification_report(
            y_test, y_pred_best, zero_division=0))

        cm = confusion_matrix(y_test, y_pred_best)
        cm_df = pd.DataFrame(cm)
        cm_df.columns = [
            f"Predicted {c}" for c in range(1, cm_df.shape[1] + 1)]
        cm_df.index = [f"Actual {r}" for r in range(1, cm_df.shape[0] + 1)]
        print("\nConfusion Matrix:\n", cm_df)

    return best_model, y_pred_best, model_comparison_df


def use_logistic_regression(x_var_df, y_var_df):
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_var_df, y_var_df, test_size=0.2, random_state=42)

    # Train logistic regression on train data and get confusion matrix on test data
    conf_matrix = _use_logistic_regression(X_train, X_test, y_train, y_test)

    # Fit logistic regression on entire dataset for summary statistics
    # Add intercept, keep DataFrame structure
    x_var_df2 = sm.add_constant(x_var_df)
    y_flat = y_var_df.values.reshape(-1) if hasattr(y_var_df,
                                                    'values') else y_var_df

    model = sm.Logit(y_flat, x_var_df2)
    results = model.fit(disp=False)  # disp=False suppresses output

    # Create summary DataFrame for coefficients and p-values
    summary_df = pd.DataFrame({
        'Coefficient': results.params,
        'p_value': results.pvalues
    })

    # Optional: process summary_df if you have this utility
    summary_df = ml_methods_utils.process_summary_df(summary_df)

    return summary_df, conf_matrix


# def use_logistic_regression(x_var_df, y_var_df):

#     # To get the confusion matrix
#     X_train, X_test, y_train, y_test = train_test_split(
#         x_var_df, y_var_df, test_size=0.2, random_state=42)

#     conf_matrix = _use_logistic_regression(X_train, X_test, y_train, y_test)

#     # Fit the model on the entire dataset for coefficients and p-values
#     model = sm.Logit(y_var_df.values.reshape(-1), x_var_df)
#     results = model.fit()
#     summary_df = pd.DataFrame(
#         {'p_value': results.pvalues, 'Coefficient': results.params})
#     summary_df = ml_methods_utils.process_summary_df(summary_df)
#     return summary_df, conf_matrix


def _use_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    results = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    return conf_matrix


def use_logistic_regression_cv(x_var_df, y_var_df, cv=5, groups=None, verbose=False):
    """
    Perform cross-validation with logistic regression.

    Parameters:
    - x_var_df: Features dataframe.
    - y_var_df: Target series or dataframe column.
    - cv: integer or cross-validation splitter object.
    - groups: array-like, group labels for samples (used if cv is GroupKFold).

    Returns:
    - dict of train/test metrics (mean ± std).
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv_results = cross_validate(
        LogisticRegression(max_iter=1000),
        x_var_df,
        y_var_df,
        cv=cv,
        scoring=scoring,
        groups=groups,
        return_train_score=True
    )

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = {}

    # Print header first
    if verbose:
        print(
            f"{'Metric':<12} {'Train Mean':>12} {'Test Mean':>12} {'Train Std':>12}   {'Test Std':>12}")
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
            print(
                f"{metric.capitalize():<12} {train_mean:12.4f} {test_mean:12.4f} {train_std:12.4f}   {test_std:12.4f}")

    return results


# def use_logistic_regression_cv(x_var_df, y_var_df, num_folds=5):
#     scoring = {
#         'accuracy': 'accuracy',
#         'precision': 'precision',
#         'recall': 'recall',
#         'f1': 'f1',
#         'roc_auc': 'roc_auc'
#     }

#     cv_results = cross_validate(
#         LogisticRegression(max_iter=1000),
#         x_var_df,
#         y_var_df,
#         cv=num_folds,
#         scoring=scoring,
#         return_train_score=True
#     )

#     metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

#     print(f"{'Metric':<12}{'Train Mean':>15} ± {'Std':<10}{'Test Mean':>15} ± {'Std':<10}")
#     print("-" * 70)

#     results = {}

#     for metric in metrics:
#         train_scores = cv_results[f'train_{metric}']
#         test_scores = cv_results[f'test_{metric}']
#         train_mean, train_std = np.mean(train_scores), np.std(train_scores)
#         test_mean, test_std = np.mean(test_scores), np.std(test_scores)

#         results[f'train_{metric}'] = (train_mean, train_std)
#         results[f'test_{metric}'] = (test_mean, test_std)

#         print(f"{metric.capitalize():<12}"
#               f"{train_mean:>15.4f} ± {train_std:<10.4f}"
#               f"{test_mean:>15.4f} ± {test_std:<10.4f}")

#     return results


# def use_logistic_regression_cv(x_var_df, y_var_df, num_folds=5):

#     cv_results = cross_validate(
#         LogisticRegression(),
#         x_var_df,
#         y_var_df,
#         cv=num_folds,
#         scoring='accuracy',
#         return_train_score=True
#     )

#     test_accuracies = cv_results['test_score']
#     train_accuracies = cv_results['train_score']

#     # Calculate the average accuracy
#     test_avg_accuracy = np.mean(test_accuracies)
#     train_avg_accuracy = np.mean(train_accuracies)

#     # print(f"Average Train Accuracy (10-fold CV): {train_avg_accuracy:.4f}")
#     # print(f"Average Test Accuracy  (10-fold CV): {test_avg_accuracy:.4f}")

#     print(
#         f"Average Accuracy (10-fold CV)\n"
#         f"  Train: {train_avg_accuracy:.4f}\n"
#         f"   Test: {test_avg_accuracy:.4f}"
#     )

#     return test_accuracies, train_accuracies, test_avg_accuracy, train_avg_accuracy


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
        self.y_value = nn.Linear(200, n_classes)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = torch.sigmoid(self.y_value(x))
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
