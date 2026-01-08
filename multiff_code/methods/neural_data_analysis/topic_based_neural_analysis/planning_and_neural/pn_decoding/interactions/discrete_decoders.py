import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score



    # ------------------------
   
MODEL_REGISTRY = {
    'logreg': 'Logistic regression (L2)',
    'svm': 'Linear SVM',
    'ridge': 'Ridge classifier',
    'lda': 'LDA',
}

from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

def make_decoder(model_type):
    if model_type == 'logreg':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=1000
            ))
        ])

    if model_type == 'svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(C=1.0))
        ])

    if model_type == 'ridge':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RidgeClassifier(alpha=1.0))
        ])

    if model_type == 'lda':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis())
        ])

    raise ValueError(f'Unknown model_type: {model_type}')


def get_feature_matrix(x_df):
    """
    Returns neural feature matrix X and feature names.
    """
    feature_cols = list(x_df.columns)
    X = x_df.values
    return X, feature_cols

def decode_behavioral_variable_xy(
    x_df,
    y_df,
    label_col,
    model_type='logreg',
    n_splits=5,
    random_state=0,
    verbose=1,
):
    """
    Decode a behavioral variable from neural data.

    Parameters
    ----------
    x_df : pandas.DataFrame
        Neural features (rows aligned with y_df)
    y_df : pandas.DataFrame
        Behavioral variables / labels
    label_col : str
        Column in y_df to decode
    """

    # ------------------------
    # Sanity checks
    # ------------------------
    assert len(x_df) == len(y_df), 'x_df and y_df must align row-wise'
    assert label_col in y_df.columns, f'{label_col} not in y_df'

    # ------------------------
    # Extract X and y
    # ------------------------
    X, feature_cols = get_feature_matrix(x_df)
    y = y_df[label_col].values

    # Remove missing labels
    valid_mask = pd.notnull(y)
    X = X[valid_mask]
    y = y[valid_mask]

    # ------------------------
    # Decoder
    # ------------------------
    clf = make_decoder(model_type)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])

        rows.append({
            'label': label_col,
            'model': model_type,
            'fold': fold,
            'accuracy': accuracy_score(y[test_idx], y_pred),
            'balanced_accuracy': balanced_accuracy_score(y[test_idx], y_pred),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })

        if verbose >= 2:
            print(f'Fold {fold} confusion matrix:')
            labels = np.unique(y)
            plot_confusion_matrix(
                y_true=y[test_idx],
                y_pred=y_pred,
                labels=labels,
                title='Speed × Angle decoding'
            )

    return pd.DataFrame(rows)


def sweep_decoders_xy(
    x_df,
    y_df,
    label_col,
    model_types=None,
    n_splits=5,
    random_state=0,
):
    if model_types is None:
        model_types = list(MODEL_REGISTRY.keys())

    all_results = []

    for model_type in model_types:
        res = decode_behavioral_variable_xy(
            x_df=x_df,
            y_df=y_df,
            label_col=label_col,
            model_type=model_type,
            n_splits=n_splits,
            random_state=random_state,
        )
        all_results.append(res)

    return pd.concat(all_results, ignore_index=True)

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC


def make_decoder_with_params(model_type, params):
    if model_type == 'logreg':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                multi_class='auto',
                max_iter=2000,
                class_weight=params.get('class_weight', None),
                C=params.get('C', 1.0),
            ))
        ])

    if model_type == 'svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(
                C=params.get('C', 1.0),
                class_weight=params.get('class_weight', None),
            ))
        ])

    if model_type == 'ridge':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RidgeClassifier(
                alpha=params.get('alpha', 1.0),
                class_weight=params.get('class_weight', None),
            ))
        ])

    raise ValueError(f'Unknown model_type: {model_type}')


def decode_with_param_sweep_xy(
    x_df,
    y_df,
    label_col,
    model_type='logreg',
    param_grid=None,
    n_splits=5,
    random_state=0,
):
    assert len(x_df) == len(y_df)
    y = y_df[label_col].astype(str)
    valid_mask = y.notna().values

    X = x_df.loc[valid_mask].values
    y = y.loc[valid_mask].values

    if param_grid is None:
        if model_type in ['logreg', 'svm']:
            param_grid = [
                {'C': 0.01, 'class_weight': 'balanced'},
                {'C': 0.1,  'class_weight': 'balanced'},
                {'C': 1.0,  'class_weight': 'balanced'},
                {'C': 10.0, 'class_weight': 'balanced'},
            ]
        elif model_type == 'ridge':
            param_grid = [
                {'alpha': 0.1,  'class_weight': 'balanced'},
                {'alpha': 1.0,  'class_weight': 'balanced'},
                {'alpha': 10.0, 'class_weight': 'balanced'},
                {'alpha': 100.0,'class_weight': 'balanced'},
            ]
        else:
            raise ValueError('Provide param_grid for this model_type')

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # pick best params using a simple internal split of train via CV-on-train
        # (lightweight: 3-fold on training data)
        inner_skf = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=random_state
        )

        best_params = None
        best_score = -np.inf

        for params in param_grid:
            inner_scores = []
            clf = make_decoder_with_params(model_type, params)

            for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
                clf.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                y_val_pred = clf.predict(X_train[inner_val_idx])
                inner_scores.append(
                    balanced_accuracy_score(y_train[inner_val_idx], y_val_pred)
                )

            mean_inner = float(np.mean(inner_scores))
            if mean_inner > best_score:
                best_score = mean_inner
                best_params = params

        # fit best on full training fold, evaluate on test fold
        clf = make_decoder_with_params(model_type, best_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        rows.append({
            'label': label_col,
            'model': model_type,
            'fold': fold,
            'best_params': str(best_params),
            'acc': accuracy_score(y_test, y_pred),
            'bal_acc': balanced_accuracy_score(y_test, y_pred),
        })

    return pd.DataFrame(rows)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    normalize=True,
    title='Confusion matrix',
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, origin='lower')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()


def compare_interaction_vs_components(
    x_df,
    y_df,
    interaction_col,
    component_cols,
    model_type='logreg',
    n_splits=5,
    min_classes=2,
):
    all_results = []

    # ------------------------
    # Interaction decoding
    # ------------------------
    if y_df[interaction_col].nunique() >= min_classes:
        res_inter = decode_behavioral_variable_xy(
            x_df=x_df,
            y_df=y_df,
            label_col=interaction_col,
            model_type=model_type,
            n_splits=n_splits,
        )
        res_inter['target'] = interaction_col
        res_inter['target_type'] = 'interaction'
        all_results.append(res_inter)

    # ------------------------
    # Component decoding
    # ------------------------
    for col in component_cols:
        # Skip if only one class (e.g. speed_band within speed_band)
        if y_df[col].nunique() < min_classes:
            continue

        res = decode_behavioral_variable_xy(
            x_df=x_df,
            y_df=y_df,
            label_col=col,
            model_type=model_type,
            n_splits=n_splits,
        )
        res['target'] = col
        res['target_type'] = 'component'
        all_results.append(res)

    if len(all_results) == 0:
        return None

    return pd.concat(all_results, ignore_index=True)


def compare_interaction_conditioned(
    x_df,
    y_df,
    interaction_col,
    component_cols,
    condition_col,
    model_type='logreg',
    n_splits=5,
):
    results = []

    for val in y_df[condition_col].cat.categories:
        mask = y_df[condition_col] == val
        if mask.sum() < 200:
            continue

        res = compare_interaction_vs_components(
            x_df.loc[mask].reset_index(drop=True),
            y_df.loc[mask].reset_index(drop=True),
            interaction_col=interaction_col,
            component_cols=component_cols,
            model_type=model_type,
            n_splits=n_splits,
        )
        res['condition_col'] = condition_col
        res['condition_value'] = val
        results.append(res)

    return pd.concat(results, ignore_index=True)

from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd

def hierarchical_decode_speed_angle(
    x_df,
    y_df,
    speed_col='speed_band',
    angle_col='cur_ff_angle_band',
    model_type='logreg',
    n_splits=5,
    random_state=0,
):
    """
    Hierarchical decoding:
    A trial is correct only if BOTH speed and angle are decoded correctly.
    """

    assert len(x_df) == len(y_df)

    X = x_df.values
    y_speed = y_df[speed_col].values
    y_angle = y_df[angle_col].values

    valid_mask = pd.notnull(y_speed) & pd.notnull(y_angle)
    X = X[valid_mask]
    y_speed = y_speed[valid_mask]
    y_angle = y_angle[valid_mask]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_speed)):
        # --- train separate decoders ---
        speed_clf = make_decoder(model_type)
        angle_clf = make_decoder(model_type)

        speed_clf.fit(X[train_idx], y_speed[train_idx])
        angle_clf.fit(X[train_idx], y_angle[train_idx])

        # --- predict ---
        speed_pred = speed_clf.predict(X[test_idx])
        angle_pred = angle_clf.predict(X[test_idx])

        # --- hierarchical correctness ---
        joint_correct = (
            (speed_pred == y_speed[test_idx]) &
            (angle_pred == y_angle[test_idx])
        )

        # balanced accuracy analogue:
        # fraction of correct trials (chance is product of marginals)
        joint_acc = joint_correct.mean()

        rows.append({
            'fold': fold,
            'joint_accuracy': joint_acc,
            'speed_accuracy': (speed_pred == y_speed[test_idx]).mean(),
            'angle_accuracy': (angle_pred == y_angle[test_idx]).mean(),
        })

    return pd.DataFrame(rows)

def cross_condition_decode(
    x_df,
    y_df,
    target_col,
    condition_col,
    train_conditions,
    test_conditions,
    model_type='logreg',
):
    """
    Train decoder on one set of condition values, test on another.
    """

    train_mask = y_df[condition_col].isin(train_conditions)
    test_mask = y_df[condition_col].isin(test_conditions)

    if train_mask.sum() < 200 or test_mask.sum() < 200:
        return None

    X_train = x_df.loc[train_mask].values
    y_train = y_df.loc[train_mask, target_col].values

    X_test = x_df.loc[test_mask].values
    y_test = y_df.loc[test_mask, target_col].values

    # ensure multiple classes
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    clf = make_decoder(model_type)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        'target': target_col,
        'condition_col': condition_col,
        'train_conditions': tuple(train_conditions),
        'test_conditions': tuple(test_conditions),
        'accuracy': (y_pred == y_test).mean(),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'n_train': len(y_train),
        'n_test': len(y_test),
    }
