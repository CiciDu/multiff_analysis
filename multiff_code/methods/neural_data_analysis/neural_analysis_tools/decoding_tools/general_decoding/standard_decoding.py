import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from neural_data_analysis.design_kits.design_by_segment import create_pn_design_df
from neural_data_analysis.topic_based_neural_analysis.planning_and_neural import pn_aligned_by_event
from data_wrangling import general_utils


# ---------------------------------------------------------------------
# Core CV decoding routine
# ---------------------------------------------------------------------
def run_population_decoder(
    X_neural,
    y,
    groups,
    decoder='logreg',
    n_splits=5
):
    """
    decoder:
        'logreg'   - linear
        'lda'      - linear
        'svm_rbf'  - nonlinear
        'rf'       - nonlinear
        'mlp'      - nonlinear (shallow, population-size aware)
    """

    gkf = GroupKFold(n_splits=n_splits)

    aucs = []
    pr_aucs = []

    n_features = X_neural.shape[1]

    for fold, (tr, te) in enumerate(gkf.split(X_neural, y, groups)):

        X_tr, X_te = X_neural[tr], X_neural[te]
        y_tr, y_te = y[tr], y[te]

        # --------------------------------------------------
        # Decoder selection
        # --------------------------------------------------
        if decoder == 'logreg':
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            clf = LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=2000,
                class_weight='balanced'
            )

        elif decoder == 'lda':
            clf = LinearDiscriminantAnalysis(solver='svd')

        elif decoder == 'svm_rbf':
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            clf = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced'
            )

        elif decoder == 'rf':
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                class_weight='balanced',
                n_jobs=-1,
                random_state=fold
            )

        elif decoder == 'mlp':
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            hidden_units = min(20, n_features)

            clf = MLPClassifier(
                hidden_layer_sizes=(hidden_units,),
                activation='relu',
                solver='adam',
                alpha=1e-3,              # stronger regularization for small N
                max_iter=2000,
                early_stopping=True,
                random_state=fold
            )

        else:
            raise ValueError(f'Unknown decoder: {decoder}')

        clf.fit(X_tr, y_tr)

        p_te = clf.predict_proba(X_te)[:, 1]

        aucs.append(roc_auc_score(y_te, p_te))
        pr_aucs.append(average_precision_score(y_te, p_te))

    return {
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'pr_mean': np.mean(pr_aucs),
        'pr_std': np.std(pr_aucs),
        'n_splits': n_splits
    }


# ---------------------------------------------------------------------
# Wrapper: decode multiple task variables
# ---------------------------------------------------------------------

def population_decoding_cv(
    cols_to_decode,
    df_X,
    df_Y,
    groups,
    decoders=(
        'logreg',   # linear
        'lda',      # linear
        'svm_rbf',  # nonlinear
        'rf',       # nonlinear
        'mlp'       # nonlinear
    ),
    n_splits=5
):

    X_neural = df_Y.to_numpy()

    for decoding_col in cols_to_decode:
        y = df_X[decoding_col].to_numpy()

        print('-' * 100)
        print(f'=== Decoding {decoding_col} from population activity ===')
        print(f'Mean label = {y.mean():.3f}')

        for dec in decoders:
            res = run_population_decoder(
                X_neural=X_neural,
                y=y,
                groups=groups,
                decoder=dec,
                n_splits=n_splits
            )

            print(
                f'{dec:12s} | '
                f'AUC = {res["auc_mean"]:.3f} ± {res["auc_std"]:.3f} | '
                f'PR-AUC = {res["pr_mean"]:.3f} ± {res["pr_std"]:.3f}'
            )


# ---------------------------------------------------------------------
# Permutation test (group-respecting)
# ---------------------------------------------------------------------

def permutation_test_auc(y, p, groups, n_perm=2000, rng=0):

    rng = np.random.default_rng(rng)
    auc_obs = roc_auc_score(y, p)

    null = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = general_utils.shuffle_within_groups(y, groups, rng)
        null[i] = roc_auc_score(y_perm, p)

    pval = (1 + np.sum(null >= auc_obs)) / (1 + n_perm)
    return auc_obs, pval, null
