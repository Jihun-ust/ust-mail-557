
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

def feature_matrix(df, use_emb=True):
    base = ["page_count","ocr_conf","image_pct","table_density","token_count_k","amount_usd","signatures","layout_complexity"]
    if use_emb:
        emb_cols = [c for c in df.columns if c.startswith("emb")]
        base = emb_cols + base
    X = df[base].values.astype(float)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    return Xs, base, sc

def pca_2d(Xs):
    from sklearn.decomposition import PCA
    p = PCA(n_components=2, random_state=42)
    return p.fit_transform(Xs), p

def tsne_2d(Xs, n=2000):
    from sklearn.manifold import TSNE
    n = min(n, Xs.shape[0])
    Xsub = Xs[:n]
    ts = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30, random_state=42)
    return ts.fit_transform(Xsub)

def try_umap_2d(Xs, n=2000):
    try:
        import umap
    except Exception:
        return None, None
    n = min(n, Xs.shape[0])
    emb = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.1, random_state=42).fit_transform(Xs[:n])
    return emb, "umap"

def k_distance_plot(Xs, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(Xs)
    dists, idx = nn.kneighbors(Xs)
    # distance to kth neighbor
    kd = np.sort(dists[:, -1])
    return kd

def anomaly_metrics(y_true, score):
    # Higher score = more anomalous
    roc = roc_auc_score(y_true, score)
    ap = average_precision_score(y_true, score)
    return {"roc_auc": float(roc), "pr_auc": float(ap)}

def plot_xy(X2, title="", labels=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    if labels is None:
        plt.scatter(X2[:,0], X2[:,1], s=10)
    else:
        # simple coloring by labels (matplotlib defaults)
        plt.scatter(X2[:,0], X2[:,1], c=labels, s=10)
    plt.title(title); plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout(); plt.show()
