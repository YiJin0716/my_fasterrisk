import os.path
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os, time, json, math
from typing import List, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier

def filter_age_le_70(df):
    return df[df['age'] <= 70]

def _norm_text(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('_','-').replace('–','-').replace('—','-')
    return s

def normalize_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Canonicalize gender to {'male','female'}
    - Canonicalize Nodule_Type to {'solid','ground-glass','part-solid'} using common aliases
    - Create binaries: sex, part_solid, ground_glass, solid
    - Coerce Upper_Lobe/Spiculation to ints if present
    - Drop rows with unmapped gender/Nodule_Type; warn on non-exclusive types
    """
    df = df.copy()

    # gender
    g = df['gender'].apply(_norm_text)
    df['gender'] = g.map({'male':'male','m':'male','female':'female','f':'female'})

    # nodule type
    t = df['Nodule_Type'].apply(_norm_text)
    mapping = {
        'solid':'solid', 'sld':'solid',

        'ground-glass':'ground-glass', 'ground glass':'ground-glass',
        'ggo':'ground-glass', 'non-solid':'ground-glass',
        'non solid':'ground-glass', 'nonsolid':'ground-glass',

        'part-solid':'part-solid', 'part solid':'part-solid',
        'semi-solid':'part-solid', 'semisolid':'part-solid', 'subsolid':'part-solid',
    }
    df['Nodule_Type'] = t.map(mapping)

    # drop unmapped
    bad = df['gender'].isna() | df['Nodule_Type'].isna()
    if bad.any():
        print(f"[filter] Dropping {int(bad.sum())} rows with unmapped gender or Nodule_Type")
        # Uncomment to inspect:
        # print(df.loc[bad, [ID_COL,'gender','Nodule_Type']].head(10))
        df = df.loc[~bad].copy()

    # binaries
    df['sex']          = df['gender'].map({'male':0, 'female':1}).astype(int)
    df['part_solid']   = (df['Nodule_Type'] == 'part-solid').astype(int)
    df['ground_glass'] = (df['Nodule_Type'] == 'ground-glass').astype(int)
    df['solid']        = (df['Nodule_Type'] == 'solid').astype(int)

    # passthrough ints
    for b in ['Upper_Lobe','Spiculation']:
        if b in df.columns:
            df[b] = pd.to_numeric(df[b], errors='ignore')
            df[b] = pd.to_numeric(df[b], errors='coerce').fillna(0).astype(int)

    # sanity: exactly one nodule type flag
    onehot_sum = df[['solid','ground_glass','part_solid']].sum(axis=1)
    if (onehot_sum != 1).any():
        print(f"[warn] {int((onehot_sum != 1).sum())} rows have non-exclusive nodule types; please inspect.")

    return df

def to_fastrisk_y(y_raw, pos_label=1) -> np.ndarray:
    """Return 1-D np.ndarray[float] with labels in {-1.0, +1.0}."""
    y_arr = np.asarray(y_raw).ravel()
    uniq = set(np.unique(y_arr))
    if uniq <= {0, 1}:
        return (2 * y_arr - 1).astype(float)
    return np.where(y_arr == pos_label, 1.0, -1.0).astype(float)

def make_ge_bins(df: pd.DataFrame, feature: str, cuts: list[float]) -> pd.DataFrame:
    """
    Build binary features like 'age >= 73' or 'sct_long_dia >= 20' (ASCII >=)
    """
    out = pd.DataFrame(index=df.index)
    vals = pd.to_numeric(df[feature], errors='coerce')
    for c in cuts:
        col = f"{feature} >= {c:g}"
        out[col] = (vals >= float(c)).astype(int)
    return out

def build_binary_matrix(X_df: pd.DataFrame,
                        feature_cuts: dict[str, list[float]],
                        passthrough_binary: list[str]) -> pd.DataFrame:
    """Construct binary feature matrix for FasterRisk from raw DataFrame"""
    mats = []
    for feat, cuts in feature_cuts.items():
        if feat not in X_df.columns:
            raise KeyError(f"Missing feature for binning: {feat}")
        mats.append(make_ge_bins(X_df, feat, cuts))
    for feat in passthrough_binary:
        if feat not in X_df.columns:
            raise KeyError(f"Missing binary feature: {feat}")
        col = pd.Series(pd.to_numeric(X_df[feat], errors='coerce')).fillna(0).astype(int)
        mats.append(pd.DataFrame({feat: col}, index=X_df.index))
    return pd.concat(mats, axis=1)

def binarize_and_align_custom(X_train_df: pd.DataFrame,
                              X_val_df: pd.DataFrame,
                              X_test_df: pd.DataFrame,
                              feature_cuts: dict[str, list[float]],
                              passthrough_binary: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Binarize each split with ≥ cuts, then align val/test to training columns.
    """
    X_train_bin = build_binary_matrix(X_train_df, feature_cuts, passthrough_binary)
    X_val_bin   = build_binary_matrix(X_val_df,   feature_cuts, passthrough_binary)
    X_test_bin  = build_binary_matrix(X_test_df,  feature_cuts, passthrough_binary)

    def _align_like_train(train_bin_df: pd.DataFrame, other_bin_df: pd.DataFrame) -> pd.DataFrame:
        cols = list(train_bin_df.columns)
        return other_bin_df.reindex(columns=cols, fill_value=0)

    X_val_bin  = _align_like_train(X_train_bin, X_val_bin)
    X_test_bin = _align_like_train(X_train_bin, X_test_bin)

    # sanity
    assert list(X_val_bin.columns)  == list(X_train_bin.columns)
    assert list(X_test_bin.columns) == list(X_train_bin.columns)
    return X_train_bin, X_val_bin, X_test_bin

def prepare_data(df: pd.DataFrame, feature_cols: list[str], label_col: str):
    X = df[feature_cols]
    y = df[label_col]
    return X, y

# ---------------------------------------------
# Version-robust optimizer creation
# ---------------------------------------------
def make_optimizer(X, y,
                   k, parent_size,
                   gap_tolerance,
                   select_top_m,
                   max_attempts,
                   want_intercept=True):
    
    # (1) gap + select_top_m + maxAttempts (camelCase)
   
        return RiskScoreOptimizer(
            X=X, y=y, k=k, parent_size=parent_size,
            gap_tolerance=gap_tolerance,
            select_top_m=select_top_m,
            maxAttempts=max_attempts)


# ---------------------------------------------
# extract models as (multipliers, beta0_int, betas_int)
# ---------------------------------------------
def extract_models(ret):
    if isinstance(ret, tuple) and len(ret) == 2:
        multipliers, integer_mat = ret
        integer_mat = np.asarray(integer_mat, dtype=int)
        beta0_int = integer_mat[:, 0]
        betas_int = integer_mat[:, 1:]
        return np.asarray(multipliers, float), beta0_int, betas_int
    elif isinstance(ret, tuple) and len(ret) == 3:
        multipliers, beta0_int, betas_int = ret
        return (np.asarray(multipliers, float),
                np.asarray(beta0_int, int),
                np.asarray(betas_int, int))
    else:
        raise RuntimeError("Unexpected return format from get_models()")


# ---------------------------------------------
# compute predicted probabilities of a single model and single data point
# ---------------------------------------------
def model_probs(mult: float, b0: float, betas: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = b0 + X @ betas.astype(float)
    return 1.0 / (1.0 + np.exp(-z/mult))


# ---------------------------------------------
# Use entire test set to calculate AUC, accuracy, and predicted prob list
# ---------------------------------------------
def compute_model_metrics(multipliers, intercepts, coef_matrix, X, y):
    aucs, accs, n_terms, probs_list = [], [], [], []
    y01 = ((y + 1.0) / 2.0)  # {-1,+1} -> {0,1}
    for m, b0, w in zip(multipliers, intercepts, coef_matrix):
        w = np.asarray(w, dtype=int)
        p = model_probs(float(m), float(b0), w, X)
        yhat = (p >= 0.5).astype(int)
        acc = (yhat == y01).mean()
        fpr, tpr, _ = roc_curve(y01, p)
        auc_val = auc(fpr, tpr)
        aucs.append(float(auc_val))
        accs.append(float(acc))
        n_terms.append(int((w != 0).sum()))
        probs_list.append(p)
    return np.asarray(aucs), np.asarray(accs), np.asarray(n_terms), probs_list


# ---------------------------------------------
# build matrix for risk score of each model
# ---------------------------------------------
def build_feature_model_matrix(
    sparseDiversePool_betas_integer,
    feature_names,
    model_prefix="model"
):
    M, p = sparseDiversePool_betas_integer.shape
    assert len(feature_names) == p

    df = pd.DataFrame(
        sparseDiversePool_betas_integer.T,
        index=feature_names,
        columns=[f"{model_prefix}_{i}" for i in range(M)]
    )
    return df


# ---------------------------------------------
# plot the whole rashomon set
# ---------------------------------------------
def plot_feature_model_matrix(
    betas_int,
    X_train_STLMD_bin,
    figsize=(18, 8),
    pos_color="#E07A5F", 
    neg_color="#4D96FF", 
    base_size=120,
    save_path=None,
    OUTDIR=None
):
    feature_model_df = build_feature_model_matrix(betas_int, feature_names=X_train_STLMD_bin.columns)
    feature_model_df = feature_model_df[(feature_model_df != 0).any(axis=1)]
    features = feature_model_df.index.tolist()
    models   = feature_model_df.columns.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    for i, feature in enumerate(features):
        for j, model in enumerate(models):
            coef = feature_model_df.loc[feature, model]

            if coef == 0:
                continue

            color = pos_color if coef > 0 else neg_color
            size  = base_size * abs(coef)

            ax.scatter(
                j, i,
                s=size,
                color=color,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.8,
                zorder=3
            )

            ax.text(
                j, i,
                f"{int(coef)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
                zorder=4
            )

    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=90)

    ax.invert_yaxis()
    ax.set_xlabel("Model (sorted by logistic loss)")
    ax.set_ylabel("Feature")

    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="-", alpha=0.25)
    ax.grid(axis="y", linestyle=":", alpha=0.2)

    plt.tight_layout()

    if OUTDIR is not None:
        if save_path is None:
            save_path = os.path.join(OUTDIR, "model.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[plot] Saved model matrix figure to {save_path}")

    
