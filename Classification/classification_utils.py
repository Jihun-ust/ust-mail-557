
import numpy as np, pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def chrono_split(df, time_col, test_frac=0.2):
    df = df.sort_values(time_col).reset_index(drop=True)
    split = int(len(df)*(1-test_frac))
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def evaluate_classifier(y_true, y_prob, threshold=0.5, title_prefix=""):
    y_pred = (y_prob >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    print(f"AUC-ROC: {roc:.3f}  |  AP (PR AUC): {pr:.3f}")
    print("Confusion matrix @ threshold", threshold, ":\n", cm)
    print("\nReport:\n", classification_report(y_true, y_pred, digits=3))
    # Curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(10,4)); plt.plot(fpr, tpr); plt.title(f"{title_prefix} ROC"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout(); plt.show()
    plt.figure(figsize=(10,4)); plt.plot(rec, prec); plt.title(f"{title_prefix} Precision-Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout(); plt.show()
    return {"roc_auc": roc, "pr_auc": pr}

def search_threshold(y_true, y_prob, weight_fn=None):
    # weight_fn(tp, fp, fn, tn) -> score to maximize
    if weight_fn is None:
        weight_fn = lambda tp, fp, fn, tn: (2*tp)/(2*tp+fp+fn+1e-9)  # F1-like
    best_t, best_s = 0.5, -1
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        s = weight_fn(tp, fp, fn, tn)
        if s > best_s: best_s, best_t = s, t
    return best_t, best_s

def calibrate_probabilities(est, X_val, y_val, method="isotonic"):
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(est, method=method, cv="prefit")
    cal.fit(X_val, y_val)
    return cal

def naive_oversample(X, y):
    # Balance classes by oversampling the minority (random with replacement)
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)
    pos_idx = np.where(y==1)[0]; neg_idx = np.where(y==0)[0]
    if len(pos_idx)==0 or len(neg_idx)==0: return X, y
    if len(pos_idx) < len(neg_idx):
        add = np.random.choice(pos_idx, size=len(neg_idx)-len(pos_idx), replace=True)
        idx = np.r_[np.arange(len(y)), add]
    else:
        add = np.random.choice(neg_idx, size=len(pos_idx)-len(neg_idx), replace=True)
        idx = np.r_[np.arange(len(y)), add]
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

def try_smote(X, y):
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res, True
    except Exception:
        return X, y, False

def focal_reweighting(y_true, y_prob, gamma=2.0, alpha_pos=0.25):
    # Return sample weights to emphasize hard positives/negatives (approx focal loss)
    eps = 1e-6
    pt = y_prob*(y_true==1) + (1 - y_prob)*(y_true==0)
    weights = (alpha_pos*(y_true==1) + (1-alpha_pos)*(y_true==0)) * ((1-pt+eps)**gamma)
    return weights
