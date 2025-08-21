import numpy as np
import matplotlib.pyplot as plt

def _extract_cv(cv):
    test = np.asarray(cv["test_corrs_by_fold"])  # (F, K)
    mean_ = np.asarray(cv.get("test_corrs_mean", test.mean(axis=0)))
    std_  = np.asarray(cv.get("test_corrs_std",  test.std(axis=0)))
    train = cv.get("train_corrs_by_fold", None)
    train = np.asarray(train) if train is not None else None
    return test, mean_, std_, train

def _align_by_components(cv1, cv2):
    t1, m1, s1, tr1 = _extract_cv(cv1)
    t2, m2, s2, tr2 = _extract_cv(cv2)
    K = min(t1.shape[1], t2.shape[1])  # align by min #components
    return (t1[:, :K], m1[:K], s1[:K], tr1[:, :K] if tr1 is not None else None,
            t2[:, :K], m2[:K], s2[:K], tr2[:, :K] if tr2 is not None else None, K)

def plot_cv_means_sd(cv1, cv2, labels=("Fit 1","Fit 2"), title="CV canonical correlations — mean ± SD"):
    t1, m1, s1, _, t2, m2, s2, _, K = _align_by_components(cv1, cv2)
    x = np.arange(1, K+1)

    plt.figure(figsize=(8,5))
    plt.errorbar(x, m1, yerr=s1, marker="o", linewidth=2, capsize=3, label=f"{labels[0]} (test)")
    plt.errorbar(x, m2, yerr=s2, marker="o", linewidth=2, capsize=3, label=f"{labels[1]} (test)")
    plt.xlabel("Canonical component")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cv_all_folds_overlay(cv1, cv2, labels=("Fit 1","Fit 2"),
                              title="CV canonical correlations — all folds (overlay)",
                              jitter=0.06, alpha_points=0.35):
    t1, m1, s1, _, t2, m2, s2, _, K = _align_by_components(cv1, cv2)
    x = np.arange(1, K+1)

    plt.figure(figsize=(8,5))
    # jittered scatter for both fits
    for f in range(t1.shape[0]):
        xs = x + np.random.uniform(-jitter, jitter, size=K)
        plt.scatter(xs, t1[f], s=18, alpha=alpha_points)
    for f in range(t2.shape[0]):
        xs = x + np.random.uniform(-jitter, jitter, size=K)
        plt.scatter(xs, t2[f], s=18, alpha=alpha_points)

    # mean lines on top
    plt.plot(x, m1, marker="o", linewidth=2, label=f"{labels[0]} mean")
    plt.plot(x, m2, marker="o", linewidth=2, label=f"{labels[1]} mean")

    plt.xlabel("Canonical component")
    plt.ylabel("Correlation (test)")
    plt.title(title)
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cv_mean_difference(cv1, cv2, labels=("Fit 1","Fit 2"),
                            title="Difference in mean test corr (Fit2 − Fit1)",
                            ci_mult=1.96):
    t1, m1, s1, _, t2, m2, s2, _, K = _align_by_components(cv1, cv2)
    x = np.arange(1, K+1)
    F1, F2 = t1.shape[0], t2.shape[0]
    diff = m2 - m1
    # pooled SE for the difference
    se = np.sqrt((s1**2 / max(1, F1)) + (s2**2 / max(1, F2)))
    ci = ci_mult * se

    plt.figure(figsize=(8,5))
    plt.errorbar(x, diff, yerr=ci, fmt="o-", linewidth=2, capsize=3)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Canonical component")
    plt.ylabel("Mean difference (test corr)")
    plt.title(title + f"  [{labels[1]} − {labels[0]}]")
    plt.xticks(x)
    plt.tight_layout()
    plt.show()

def plot_cv_train_test_gap(cv1, cv2, labels=("Fit 1","Fit 2"),
                           title="Train–test gap (mean(train) − mean(test))"):
    t1, m1, s1, tr1, t2, m2, s2, tr2, K = _align_by_components(cv1, cv2)
    x = np.arange(1, K+1)

    gaps = []
    lbls = []
    if tr1 is not None:
        gaps.append(tr1.mean(axis=0) - t1.mean(axis=0))
        lbls.append(f"{labels[0]}")
    if tr2 is not None:
        gaps.append(tr2.mean(axis=0) - t2.mean(axis=0))
        lbls.append(f"{labels[1]}")

    if not gaps:
        print("No training correlations in one or both cv summaries; skipping train–test gap plot.")
        return

    plt.figure(figsize=(8,5))
    for g, lab in zip(gaps, lbls):
        plt.plot(x, g, marker="o", linewidth=2, label=lab)

    plt.xlabel("Canonical component")
    plt.ylabel("Mean(train) − Mean(test)")
    plt.title(title)
    plt.xticks(x)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_cv_cca_compare_all(cv1, cv2, labels=("Fit 1","Fit 2")):
    plot_cv_means_sd(cv1, cv2, labels,
                     title="CV canonical correlations — mean ± SD")
    plot_cv_all_folds_overlay(cv1, cv2, labels,
                              title="CV canonical correlations — all folds (overlay)")
    plot_cv_mean_difference(cv1, cv2, labels,
                            title="Difference in mean test corr (Fit2 − Fit1)")
    plot_cv_train_test_gap(cv1, cv2, labels,
                           title="Train–test gap (mean(train) − mean(test))")
