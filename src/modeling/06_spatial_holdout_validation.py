"""
Milestone 6: Spatial holdout validation for spatial niche classifiers.

Purpose
-------
This script evaluates whether spatial niche classifiers generalize to unseen
tissue regions.

The previous ML script used a random train/test split. That is useful as a
baseline, but random spot-level splitting can overestimate performance in
spatial transcriptomics because nearby spots are spatially correlated.

This script performs a stricter validation:

    1. Divide the tissue into spatial blocks.
    2. Hold out one complete spatial block at a time.
    3. Train models on the remaining blocks.
    4. Test models on the held-out tissue region.
    5. Repeat across valid blocks, models, and feature sets.

Why this matters
----------------
In spatial transcriptomics, neighboring spots often share:

    - similar expression profiles
    - similar tissue morphology
    - similar cell-type composition
    - similar technical effects
    - the same local biological niche

A random split can place neighboring spots from the same tissue region into
both training and testing sets. That can make performance look better than it
would be when the model sees a genuinely new tissue region.

Spatial holdout validation asks a harder question:

    Can the model classify biological niches in a tissue region it did not
    train on?

Input
-----
Final labeled AnnData object:

    data/processed/visium_human_breast_cancer_final_labeled.h5ad

Expected fields:

    adata.obs["ml_training_label"]
    adata.obs["manual_niche_label_short"]
    adata.obsm["X_pca"]
    adata.obsm["spatial"]
    adata.obsp["spatial_connectivities"]
    adata.raw

Outputs
-------
Tables:

    results/tables/spatial_block_label_counts.csv
    results/tables/spatial_holdout_validation_model_feature_metrics.csv
    results/tables/spatial_holdout_summary_by_model.csv
    results/tables/spatial_holdout_summary_by_feature_set.csv
    results/tables/spatial_holdout_summary_by_model_and_feature_set.csv
    results/tables/spatial_holdout_*_classification_report.csv

Figures:

    results/figures/06_spatial_holdout_validation/

Models tested
-------------
1. logistic_regression
2. calibrated_linear_svm
3. random_forest
4. extra_trees
5. hist_gradient_boosting

Feature sets tested
-------------------
1. expression_qc_marker
    PCA + QC + marker scores

2. expression_qc_marker_spatial
    PCA + QC + marker scores + spatial_x/spatial_y

3. expression_qc_marker_neighbor
    PCA + QC + marker scores + Squidpy neighbor marker scores

4. full_spatial_context
    PCA + QC + marker scores + spatial_x/spatial_y
    + Squidpy neighbor marker scores

Scientific caution
------------------
This is still a single-sample validation. Spatial holdout validation is more
realistic than a random split, but true generalization requires validation on
additional tissue sections or patients.
"""

from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_final_labeled.h5ad"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "06_spatial_holdout_validation"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset and label metadata
# ---------------------------------------------------------------------
LIBRARY_ID = "Visium_Human_Breast_Cancer"

LABEL_KEY = "ml_training_label"
EXCLUDE_LABEL = "Exclude_low_confidence"

SPATIAL_BLOCK_KEY = "spatial_block_3x3"
SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"


# ---------------------------------------------------------------------
# Marker sets used as interpretable features
# ---------------------------------------------------------------------
MARKER_SETS = {
    "tumor_epithelial": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "KRT7",
        "MUC1",
        "TACSTD2",
    ],
    "stromal_caf": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "ACTA2",
        "TAGLN",
    ],
    "pan_immune": [
        "PTPRC",
        "LCP1",
        "CD52",
        "CORO1A",
        "CXCL13",
    ],
    "t_cell": [
        "CD3D",
        "CD3E",
        "CD2",
        "TRAC",
        "CD8A",
        "CD8B",
        "GZMB",
        "NKG7",
    ],
    "b_cell_plasma": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
        "MZB1",
        "JCHAIN",
        "IGKC",
        "IGHG1",
        "IGHG3",
    ],
    "myeloid_apc": [
        "CD74",
        "HLA-DRA",
        "HLA-DPA1",
        "HLA-DPB1",
        "C1QA",
        "C1QB",
        "LYZ",
        "LST1",
    ],
    "endothelial": [
        "PECAM1",
        "VWF",
        "KDR",
        "ENG",
        "PLVAP",
    ],
    "proliferation": [
        "MKI67",
        "TOP2A",
        "PCNA",
        "MCM5",
        "UBE2C",
    ],
    "adipocyte_fat": [
        "FABP4",
        "PLIN1",
        "ADIPOQ",
        "LPL",
        "G0S2",
        "CFD",
    ],
    "hypoxia_glycolysis": [
        "GAPDH",
        "PGK1",
        "TPI1",
        "ENO1",
        "LDHA",
        "VEGFA",
        "CA9",
    ],
    "luminal_secretory": [
        "SCGB2A2",
        "SCGB1D2",
        "CSTA",
        "S100G",
        "GATA3",
        "XBP1",
    ],
}


def safe_name(text: str) -> str:
    """
    Convert arbitrary text into a filename-safe string.

    Why this helper exists
    ----------------------
    Feature set names, model names, and block IDs are used in output filenames.
    This function prevents spaces, slashes, punctuation, or special characters
    from creating messy filenames.
    """
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_")


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.

    Why this helper exists
    ----------------------
    This script creates many figures: spatial block maps, confusion matrices,
    and summary plots. Centralized figure saving keeps export settings
    consistent.
    """
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def plot_spatial_scatter(
    adata,
    color,
    filename: str,
    img: bool = True,
    legend_loc: str | None = "right margin",
    ncols: int = 2,
) -> None:
    """
    Plot spatial data with Squidpy and save the figure.

    Important detail
    ----------------
    In this project environment, `sq.pl.spatial_scatter()` should not be called
    with `show=False`, because unsupported arguments may be forwarded to
    Matplotlib.
    """
    kwargs = {
        "adata": adata,
        "color": color,
        "library_id": LIBRARY_ID,
        "img": img,
        "ncols": ncols,
    }

    if legend_loc is not None:
        kwargs["legend_loc"] = legend_loc

    sq.pl.spatial_scatter(**kwargs)
    save_current_fig(filename)


def validate_inputs(adata) -> None:
    """
    Check that required fields exist before spatial holdout validation.

    Why this function exists
    ------------------------
    It is better to fail early with a clear message than to fail deep inside
    scikit-learn or plotting code.
    """
    required_obs = [
        LABEL_KEY,
        "manual_niche_label_short",
    ]

    missing_obs = [col for col in required_obs if col not in adata.obs.columns]
    if missing_obs:
        raise ValueError(f"Missing required adata.obs columns: {missing_obs}")

    if "X_pca" not in adata.obsm:
        raise ValueError("Missing adata.obsm['X_pca'].")

    if "spatial" not in adata.obsm:
        raise ValueError("Missing adata.obsm['spatial'].")

    if adata.raw is None:
        warnings.warn(
            "adata.raw is missing. Marker scoring will use adata.var_names only.",
            UserWarning,
        )

    if SPATIAL_CONNECTIVITY_KEY not in adata.obsp:
        warnings.warn(
            f"{SPATIAL_CONNECTIVITY_KEY} not found. "
            "Squidpy neighbor marker features will be skipped.",
            UserWarning,
        )


def get_available_genes(adata, genes: list[str]) -> list[str]:
    """
    Return marker genes present in the AnnData object.

    Why use adata.raw?
    ------------------
    The processed object is subset to highly variable genes, while `adata.raw`
    preserves the full normalized/log-transformed expression matrix before HVG
    subsetting. Marker genes are not guaranteed to be highly variable, so raw is
    preferred when available.
    """
    if adata.raw is not None:
        available = set(adata.raw.var_names)
    else:
        available = set(adata.var_names)

    return [gene for gene in genes if gene in available]


def add_marker_signature_scores(adata):
    """
    Add marker signature score columns to adata.obs.

    Why this is included
    --------------------
    Spatial holdout validation should evaluate interpretable biological features
    in addition to PCA expression features. Marker scores capture programs such
    as tumor epithelial, immune, myeloid/APC, hypoxia/glycolysis, and luminal
    secretory signal.

    Alternative feature choices
    ---------------------------
    Future versions could use pathway scores, raw marker gene expression,
    deconvolution proportions, or image-derived morphology features.
    """
    score_columns = []

    for signature_name, genes in MARKER_SETS.items():
        available_genes = get_available_genes(adata, genes)

        print(f"\nSignature: {signature_name}")
        print(f"  Available genes: {available_genes}")

        if len(available_genes) < 2:
            print("  Skipping: fewer than 2 genes available.")
            continue

        score_col = f"score_{signature_name}"

        sc.tl.score_genes(
            adata,
            gene_list=available_genes,
            score_name=score_col,
            use_raw=True if adata.raw is not None else False,
        )

        score_columns.append(score_col)

    return adata, score_columns


def add_spatial_blocks(
    adata,
    n_bins_x: int = 3,
    n_bins_y: int = 3,
):
    """
    Divide the tissue into rectangular spatial blocks.

    Why 3 x 3?
    ----------
    A 3 x 3 grid is a practical balance:
        - enough blocks to evaluate multiple tissue regions
        - each block is large enough to contain enough test spots
        - output plots remain interpretable

    Alternative choices
    -------------------
    2 x 2 blocks create larger test regions but fewer folds.
    4 x 4 or 5 x 5 blocks create more folds but may produce sparse labels.
    """
    spatial = adata.obsm["spatial"]

    x = spatial[:, 0]
    y = spatial[:, 1]

    x_bins = pd.cut(
        x,
        bins=n_bins_x,
        labels=False,
        include_lowest=True,
    )

    y_bins = pd.cut(
        y,
        bins=n_bins_y,
        labels=False,
        include_lowest=True,
    )

    block_labels = [
        f"x{int(x_bin)}_y{int(y_bin)}"
        for x_bin, y_bin in zip(x_bins, y_bins)
    ]

    adata.obs[SPATIAL_BLOCK_KEY] = pd.Categorical(block_labels)

    return adata


def make_neighbor_score_features(
    adata,
    score_columns: list[str],
) -> pd.DataFrame:
    """
    Use the Squidpy spatial graph to calculate neighbor-averaged marker scores.

    For each marker score column:

        neighbor_mean_score_X = average score_X among spatial neighbors

    Why this matters
    ----------------
    A spot's niche may depend not only on its own expression, but also on its
    local tissue context. These features represent local biological context
    derived from the Squidpy spatial-neighbor graph.
    """
    if SPATIAL_CONNECTIVITY_KEY not in adata.obsp or not score_columns:
        return pd.DataFrame(index=adata.obs_names)

    print("\nCreating Squidpy spatial-neighborhood score features...")

    graph = adata.obsp[SPATIAL_CONNECTIVITY_KEY].copy()

    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0

    graph_normalized = graph.multiply(1.0 / row_sums[:, None])

    score_matrix = adata.obs[score_columns].astype(float).to_numpy()
    neighbor_scores = graph_normalized.dot(score_matrix)

    neighbor_columns = [f"neighbor_mean_{col}" for col in score_columns]

    neighbor_df = pd.DataFrame(
        neighbor_scores,
        index=adata.obs_names,
        columns=neighbor_columns,
    )

    print(f"  Added {neighbor_df.shape[1]} neighborhood features.")

    return neighbor_df


def build_feature_sets(
    adata,
    score_columns: list[str],
    n_pcs: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    Build multiple feature sets for spatial holdout validation.

    Feature sets
    ------------
    expression_qc_marker:
        PCA + QC + marker scores

    expression_qc_marker_spatial:
        PCA + QC + marker scores + spatial coordinates

    expression_qc_marker_neighbor:
        PCA + QC + marker scores + Squidpy neighbor marker scores

    full_spatial_context:
        PCA + QC + marker scores + spatial coordinates
        + Squidpy neighbor marker scores

    Why compare these?
    ------------------
    This lets us test whether spatial holdout performance is driven by
    expression biology, raw tissue coordinates, local spatial neighborhood
    context, or all features together.
    """
    pcs = adata.obsm["X_pca"][:, :n_pcs]
    pc_cols = [f"PC{i + 1}" for i in range(pcs.shape[1])]

    pc_df = pd.DataFrame(
        pcs,
        index=adata.obs_names,
        columns=pc_cols,
    )

    possible_qc_cols = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb",
        "pct_counts_in_top_50_genes",
        "pct_counts_in_top_100_genes",
        "pct_counts_in_top_200_genes",
    ]

    qc_cols = [col for col in possible_qc_cols if col in adata.obs.columns]

    if not qc_cols:
        raise ValueError("No QC columns found for feature construction.")

    qc_df = adata.obs[qc_cols].astype(float).copy()

    spatial_df = pd.DataFrame(
        adata.obsm["spatial"],
        index=adata.obs_names,
        columns=["spatial_x", "spatial_y"],
    )

    signature_df = adata.obs[score_columns].astype(float).copy()

    neighbor_df = make_neighbor_score_features(
        adata=adata,
        score_columns=score_columns,
    )

    base_features = pd.concat(
        [
            pc_df,
            qc_df,
            signature_df,
        ],
        axis=1,
    )

    feature_sets = {
        "expression_qc_marker": base_features,
        "expression_qc_marker_spatial": pd.concat(
            [
                base_features,
                spatial_df,
            ],
            axis=1,
        ),
        "expression_qc_marker_neighbor": pd.concat(
            [
                base_features,
                neighbor_df,
            ],
            axis=1,
        ),
        "full_spatial_context": pd.concat(
            [
                base_features,
                spatial_df,
                neighbor_df,
            ],
            axis=1,
        ),
    }

    cleaned_feature_sets = {}

    for name, features in feature_sets.items():
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median(numeric_only=True))
        cleaned_feature_sets[name] = features

        print(f"Feature set {name}: {features.shape[1]} features")

    return cleaned_feature_sets
    
def define_models() -> dict:
    """
    Define the five classifiers used for spatial holdout validation.

    Why use the same model family set as script 5?
    ---------------------------------------------
    Script 5 compared these models under a random train/test split. This script
    repeats that comparison under a stricter spatial holdout design.

    This helps answer:

        Did a model perform well because random train/test spots were spatially
        mixed, or can it generalize to held-out tissue regions?

    Models
    ------
    logistic_regression:
        Linear probabilistic baseline.

    calibrated_linear_svm:
        Linear margin-based classifier with calibrated probabilities.

    random_forest:
        Nonlinear bagged tree ensemble.

    extra_trees:
        More randomized tree ensemble.

    hist_gradient_boosting:
        Boosted tree classifier that performed best in the random-split
        baseline.

    Runtime note
    ------------
    This script trains many models:

        models x feature sets x spatial blocks

    If runtime becomes too long, reduce the model dictionary temporarily to only
    one or two models.
    """
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    calibrated_linear_svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                    ),
                    cv=3,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=3,
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=3,
    )

    hist_gradient_boosting = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=42,
    )

    return {
        "logistic_regression": logistic_regression,
        "calibrated_linear_svm": calibrated_linear_svm,
        "random_forest": random_forest,
        "extra_trees": extra_trees,
        "hist_gradient_boosting": hist_gradient_boosting,
    }


def summarize_block_labels(adata) -> None:
    """
    Save a table of trainable label counts per spatial block.

    Why this matters
    ----------------
    Spatial holdout performance depends heavily on which labels appear in each
    held-out block.

    Some blocks may contain many niche classes, while others may contain only
    one or two. A low macro F1 score may reflect a difficult block composition
    rather than simply a poor model.

    This table helps interpret:
        - block-specific model failures
        - missing labels in training or testing
        - whether some niches are spatially localized
    """
    trainable_mask = adata.obs[LABEL_KEY].astype(str) != EXCLUDE_LABEL

    block_label_counts = pd.crosstab(
        adata.obs.loc[trainable_mask, SPATIAL_BLOCK_KEY].astype(str),
        adata.obs.loc[trainable_mask, LABEL_KEY].astype(str),
    )

    block_label_counts_path = TABLE_DIR / "spatial_block_label_counts.csv"
    block_label_counts.to_csv(block_label_counts_path)

    print(f"\nSaved spatial block label counts to: {block_label_counts_path}")
    print("\nSpatial block label counts:")
    print(block_label_counts)


def plot_spatial_blocks(adata) -> None:
    """
    Plot spatial block assignments and ML training labels.

    Why this plot is useful
    -----------------------
    Before interpreting spatial holdout metrics, we should confirm what the
    held-out regions look like on the tissue.

    This figure helps answer:

        - Are blocks covering the tissue as expected?
        - Are labels unevenly distributed across the tissue?
        - Are some classes localized to specific regions?
        - Are low-confidence labels concentrated in one area?

    Squidpy is used for spatial plotting so this script is consistent with the
    updated Squidpy workflow.
    """
    plot_spatial_scatter(
        adata=adata,
        color=[SPATIAL_BLOCK_KEY, LABEL_KEY],
        filename="spatial_blocks_and_training_labels.png",
        img=True,
        legend_loc="right margin",
        ncols=2,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    filename: str,
) -> None:
    """
    Plot a raw-count confusion matrix for one holdout fold.

    Why this plot is included
    -------------------------
    Metrics summarize model performance, but confusion matrices show which
    biological niches are confused in each held-out region.

    Biological interpretation
    -------------------------
    Confusing related tumor epithelial states may be less surprising than
    confusing immune-rich and tumor-rich regions. The confusion matrix makes
    these error patterns visible.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
            )

    fig.colorbar(im, ax=ax)
    save_current_fig(filename)


def should_skip_holdout_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    heldout_block: str,
    min_test_spots: int = 30,
    min_test_labels: int = 2,
) -> bool:
    """
    Decide whether a spatial holdout fold should be skipped.

    Why folds may be skipped
    ------------------------
    Some spatial blocks may have too few trainable spots after excluding
    low-confidence labels. Others may contain only one class.

    Very small or single-class folds are not very informative for multiclass
    spatial validation.

    Current thresholds
    ------------------
    min_test_spots = 30:
        Skip test blocks with fewer than 30 trainable spots.

    min_test_labels = 2:
        Skip test blocks with fewer than two labels.

    Alternative thresholds
    ----------------------
    More permissive:
        min_test_spots = 20

    More conservative:
        min_test_spots = 50
        min_test_labels = 3
    """
    if X_test.shape[0] < min_test_spots:
        print(
            f"  Skipping {heldout_block}: fewer than "
            f"{min_test_spots} test spots."
        )
        return True

    if y_test.nunique() < min_test_labels:
        print(
            f"  Skipping {heldout_block}: fewer than "
            f"{min_test_labels} labels in test block."
        )
        return True

    if y_train.nunique() < 2:
        print(f"  Skipping {heldout_block}: fewer than 2 labels in training set.")
        return True

    return False


def train_and_evaluate_holdout(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    blocks: pd.Series,
    heldout_block: str,
    feature_set_name: str,
):
    """
    Train one model on all blocks except one and test on the held-out block.

    This is the core spatial holdout unit.

    Parameters
    ----------
    model_name:
        Name of the classifier being evaluated.

    model:
        scikit-learn classifier or pipeline.

    X:
        Feature table for trainable spots.

    y:
        Manual ML training labels for trainable spots.

    blocks:
        Spatial block assignment for each trainable spot.

    heldout_block:
        Block to hold out for testing.

    feature_set_name:
        Name of the feature set being evaluated.

    Returns
    -------
    dict or None:
        Metrics for this fold, or None if the fold was skipped.

    Important interpretation
    ------------------------
    If a held-out block contains a label that is absent from the training
    blocks, the model cannot learn that class in the fold. The script records
    those labels in `labels_missing_from_train`.
    """
    test_mask = blocks == heldout_block
    train_mask = ~test_mask

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    if should_skip_holdout_fold(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        heldout_block=heldout_block,
    ):
        return None

    train_labels = set(y_train.unique())
    test_labels = set(y_test.unique())

    labels_missing_from_train = sorted(test_labels - train_labels)

    # Fit model on all non-held-out blocks.
    model.fit(X_train, y_train)

    # Predict held-out region.
    y_pred = pd.Series(
        model.predict(X_test),
        index=y_test.index,
        name="predicted_label",
    )

    present_labels = sorted(set(y_test.unique()).union(set(y_pred.unique())))

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=present_labels,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "model": model_name,
        "feature_set": feature_set_name,
        "heldout_block": heldout_block,
        "n_train_spots": int(X_train.shape[0]),
        "n_test_spots": int(X_test.shape[0]),
        "n_train_labels": int(y_train.nunique()),
        "n_test_labels": int(y_test.nunique()),
        "test_labels": ";".join(sorted(y_test.unique())),
        "labels_missing_from_train": ";".join(labels_missing_from_train),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    print("  Metrics:")
    print(f"    model:             {model_name}")
    print(f"    feature set:       {feature_set_name}")
    print(f"    test spots:        {X_test.shape[0]}")
    print(f"    test labels:       {y_test.nunique()}")
    print(f"    accuracy:          {accuracy:.4f}")
    print(f"    balanced_accuracy: {balanced_accuracy:.4f}")
    print(f"    macro_f1:          {macro_f1:.4f}")
    print(f"    weighted_f1:       {weighted_f1:.4f}")

    if labels_missing_from_train:
        print(f"    labels missing from train: {labels_missing_from_train}")

    report = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = (
        TABLE_DIR
        / (
            "spatial_holdout_"
            f"{safe_name(model_name)}_"
            f"{safe_name(feature_set_name)}_"
            f"{safe_name(heldout_block)}_classification_report.csv"
        )
    )
    report_df.to_csv(report_path)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=present_labels,
    )

    plot_confusion_matrix(
        cm=cm,
        labels=present_labels,
        title=f"{model_name} | {feature_set_name} | held out {heldout_block}",
        filename=(
            "spatial_holdout_"
            f"{safe_name(model_name)}_"
            f"{safe_name(feature_set_name)}_"
            f"{safe_name(heldout_block)}_confusion_matrix.png"
        ),
    )

    return metrics
    
def run_spatial_holdout_validation(
    feature_sets: dict[str, pd.DataFrame],
    y: pd.Series,
    blocks: pd.Series,
) -> pd.DataFrame:
    """
    Run leave-one-spatial-block-out validation.

    This evaluates every model across every feature set and every valid spatial
    block.

    Design
    ------
    For each feature set:
        For each model:
            For each spatial block:
                Train on all other blocks.
                Test on the held-out block.

    Why this matters
    ----------------
    This lets us compare:

        - model family effects
        - feature set effects
        - tissue-region difficulty
        - whether spatial context improves or hurts generalization

    Runtime note
    ------------
    This can train many models:

        5 models x 4 feature sets x up to 9 blocks = up to 180 fits

    This is still reasonable for this dataset, but it is more intensive than
    the random split baseline.
    """
    all_metrics = []

    unique_blocks = sorted(blocks.unique())
    model_names = list(define_models().keys())

    for feature_set_name, X in feature_sets.items():
        print("\n==============================")
        print(f"Feature set: {feature_set_name}")
        print(f"Number of features: {X.shape[1]}")
        print("==============================")

        for model_name in model_names:
            print("\n------------------------------")
            print(f"Model: {model_name}")
            print("------------------------------")

            for heldout_block in unique_blocks:
                print(f"\nRunning holdout block: {heldout_block}")

                # Create a fresh model object for each fold so no fitted state
                # leaks between spatial holdout evaluations.
                model = define_models()[model_name]

                metrics = train_and_evaluate_holdout(
                    model_name=model_name,
                    model=model,
                    X=X,
                    y=y,
                    blocks=blocks,
                    heldout_block=heldout_block,
                    feature_set_name=feature_set_name,
                )

                if metrics is not None:
                    all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = TABLE_DIR / "spatial_holdout_validation_model_feature_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nSaved spatial holdout metrics to: {metrics_path}")

    return metrics_df


def save_holdout_summary_tables(metrics_df: pd.DataFrame) -> None:
    """
    Save summary tables for spatial holdout validation.

    Summary tables
    --------------
    spatial_holdout_summary_by_model.csv:
        Average performance for each model across all feature sets and blocks.

    spatial_holdout_summary_by_feature_set.csv:
        Average performance for each feature set across all models and blocks.

    spatial_holdout_summary_by_model_and_feature_set.csv:
        Average performance for each model-feature combination.

    Why these tables are useful
    ---------------------------
    The full metrics table is detailed but large. Summary tables make it easier
    to report the key findings in the README and LinkedIn/GitHub portfolio
    summary.
    """
    if metrics_df.empty:
        print("No holdout metrics available for summary tables.")
        return

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    summary_by_model = (
        metrics_df.groupby("model")[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )

    summary_by_model_path = TABLE_DIR / "spatial_holdout_summary_by_model.csv"
    summary_by_model.to_csv(summary_by_model_path)
    print(f"Saved summary by model to: {summary_by_model_path}")

    summary_by_feature_set = (
        metrics_df.groupby("feature_set")[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )

    summary_by_feature_set_path = (
        TABLE_DIR / "spatial_holdout_summary_by_feature_set.csv"
    )
    summary_by_feature_set.to_csv(summary_by_feature_set_path)
    print(f"Saved summary by feature set to: {summary_by_feature_set_path}")

    summary_by_model_feature = (
        metrics_df.groupby(["model", "feature_set"])[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )

    summary_by_model_feature_path = (
        TABLE_DIR / "spatial_holdout_summary_by_model_and_feature_set.csv"
    )
    summary_by_model_feature.to_csv(summary_by_model_feature_path)
    print(
        "Saved summary by model and feature set to: "
        f"{summary_by_model_feature_path}"
    )

    print("\nSpatial holdout summary by model:")
    print(summary_by_model)

    print("\nSpatial holdout summary by feature set:")
    print(summary_by_feature_set)

    print("\nSpatial holdout summary by model and feature set:")
    print(summary_by_model_feature)


def plot_macro_f1_by_model(metrics_df: pd.DataFrame) -> None:
    """
    Plot mean spatial holdout macro F1 by model.

    Why macro F1?
    -------------
    Macro F1 treats each class equally, which is useful because spatial niche
    classes are imbalanced.

    This plot helps answer:

        Which model generalizes best to unseen tissue regions?
    """
    if metrics_df.empty:
        return

    summary = (
        metrics_df.groupby("model")["macro_f1"]
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    summary.plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean macro F1 across spatial holdout folds")
    ax.set_ylabel("Model")
    ax.set_title("Spatial holdout validation: mean macro F1 by model")
    ax.set_xlim(0, 1.05)
    save_current_fig("spatial_holdout_macro_f1_by_model.png")


def plot_macro_f1_by_feature_set(metrics_df: pd.DataFrame) -> None:
    """
    Plot mean spatial holdout macro F1 by feature set.

    This plot helps answer:

        Do spatial coordinates or Squidpy neighbor features improve
        generalization to held-out tissue regions?
    """
    if metrics_df.empty:
        return

    summary = (
        metrics_df.groupby("feature_set")["macro_f1"]
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    summary.plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean macro F1 across spatial holdout folds")
    ax.set_ylabel("Feature set")
    ax.set_title("Spatial holdout validation: mean macro F1 by feature set")
    ax.set_xlim(0, 1.05)
    save_current_fig("spatial_holdout_macro_f1_by_feature_set.png")


def plot_macro_f1_by_model_and_feature_set(metrics_df: pd.DataFrame) -> None:
    """
    Plot model-feature-set spatial holdout performance.

    This figure shows the most complete comparison:

        model x feature set -> mean macro F1

    Why this is useful
    ------------------
    A model may only perform well with a specific feature set. This plot helps
    identify whether performance is mostly driven by the model family, the
    feature engineering strategy, or both.
    """
    if metrics_df.empty:
        return

    summary = (
        metrics_df.groupby(["model", "feature_set"])["macro_f1"]
        .mean()
        .reset_index()
    )

    summary["model_feature"] = (
        summary["model"] + " | " + summary["feature_set"]
    )

    summary = summary.sort_values("macro_f1", ascending=True)

    fig_height = max(8, 0.35 * summary.shape[0])
    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.barh(summary["model_feature"], summary["macro_f1"])
    ax.set_xlabel("Mean macro F1 across spatial holdout folds")
    ax.set_ylabel("Model | Feature set")
    ax.set_title("Spatial holdout validation: model and feature-set comparison")
    ax.set_xlim(0, 1.05)

    save_current_fig("spatial_holdout_macro_f1_by_model_and_feature_set.png")


def plot_macro_f1_by_block(metrics_df: pd.DataFrame) -> None:
    """
    Plot macro F1 across held-out spatial blocks.

    Why this plot matters
    ---------------------
    Spatial holdout performance may vary substantially across tissue regions.
    Some blocks may be easier because they contain clear, homogeneous niches.
    Others may be harder because they contain mixed tissue, rare labels, or
    boundaries between niches.
    """
    if metrics_df.empty:
        return

    # Average across models and feature sets for a tissue-region-level summary.
    block_summary = (
        metrics_df.groupby("heldout_block")["macro_f1"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    block_summary.plot(kind="bar", ax=ax)
    ax.set_xlabel("Held-out spatial block")
    ax.set_ylabel("Mean macro F1")
    ax.set_title("Spatial holdout validation: mean macro F1 by held-out block")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")

    save_current_fig("spatial_holdout_macro_f1_by_block.png")


def plot_feature_set_lines_by_block(metrics_df: pd.DataFrame) -> None:
    """
    Plot macro F1 by held-out block for each feature set.

    This plot helps identify whether a feature set performs consistently across
    tissue regions or only helps in specific blocks.
    """
    if metrics_df.empty:
        return

    # Average across models within each feature set and block.
    plot_df = (
        metrics_df.groupby(["feature_set", "heldout_block"])["macro_f1"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for feature_set, sub_df in plot_df.groupby("feature_set"):
        sub_df = sub_df.sort_values("heldout_block")
        ax.plot(
            sub_df["heldout_block"],
            sub_df["macro_f1"],
            marker="o",
            label=feature_set,
        )

    ax.set_xlabel("Held-out spatial block")
    ax.set_ylabel("Mean macro F1")
    ax.set_title("Spatial holdout validation by feature set and block")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")

    save_current_fig("spatial_holdout_macro_f1_by_feature_set_and_block.png")


def plot_model_lines_by_block(metrics_df: pd.DataFrame) -> None:
    """
    Plot macro F1 by held-out block for each model.

    This plot helps identify whether a model is consistently robust across
    tissue regions or whether it fails on particular blocks.
    """
    if metrics_df.empty:
        return

    # Average across feature sets within each model and block.
    plot_df = (
        metrics_df.groupby(["model", "heldout_block"])["macro_f1"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, sub_df in plot_df.groupby("model"):
        sub_df = sub_df.sort_values("heldout_block")
        ax.plot(
            sub_df["heldout_block"],
            sub_df["macro_f1"],
            marker="o",
            label=model,
        )

    ax.set_xlabel("Held-out spatial block")
    ax.set_ylabel("Mean macro F1")
    ax.set_title("Spatial holdout validation by model and block")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")

    save_current_fig("spatial_holdout_macro_f1_by_model_and_block.png")


def plot_holdout_summary_figures(metrics_df: pd.DataFrame) -> None:
    """
    Generate all spatial holdout summary plots.

    These plots are designed for interpretation and README reporting.
    """
    if metrics_df.empty:
        print("No holdout metrics available to plot.")
        return

    plot_macro_f1_by_model(metrics_df)
    plot_macro_f1_by_feature_set(metrics_df)
    plot_macro_f1_by_model_and_feature_set(metrics_df)
    plot_macro_f1_by_block(metrics_df)
    plot_feature_set_lines_by_block(metrics_df)
    plot_model_lines_by_block(metrics_df)


def main() -> None:
    """
    Run spatial holdout validation.

    Workflow
    --------
    1. Load final labeled AnnData object.
    2. Validate required fields.
    3. Add marker signature scores.
    4. Divide tissue into a 3 x 3 spatial grid.
    5. Plot spatial blocks and training labels.
    6. Summarize label composition by block.
    7. Build four feature sets.
    8. Exclude low-confidence labels.
    9. Run leave-one-spatial-block-out validation for five models.
    10. Save fold-level metrics, summary tables, reports, and plots.
    """
    print(f"Loading final labeled AnnData object from: {INPUT_H5AD}")

    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded object:")
    print(adata)

    validate_inputs(adata)

    print("\nOriginal ML label counts:")
    print(adata.obs[LABEL_KEY].astype(str).value_counts())

    # ------------------------------------------------------------------
    # 1. Add marker signature scores
    # ------------------------------------------------------------------
    adata, score_columns = add_marker_signature_scores(adata)

    print("\nMarker signature score columns:")
    print(score_columns)

    # ------------------------------------------------------------------
    # 2. Add spatial blocks
    # ------------------------------------------------------------------
    adata = add_spatial_blocks(
        adata,
        n_bins_x=3,
        n_bins_y=3,
    )

    print("\nSpatial block counts:")
    print(adata.obs[SPATIAL_BLOCK_KEY].astype(str).value_counts().sort_index())

    summarize_block_labels(adata)
    plot_spatial_blocks(adata)

    # ------------------------------------------------------------------
    # 3. Build feature sets
    # ------------------------------------------------------------------
    feature_sets_all = build_feature_sets(
        adata=adata,
        score_columns=score_columns,
        n_pcs=30,
    )

    # Save feature column lists for transparency.
    feature_column_rows = []

    for feature_set_name, X in feature_sets_all.items():
        for feature in X.columns:
            feature_column_rows.append(
                {
                    "feature_set": feature_set_name,
                    "feature": feature,
                }
            )

    feature_columns_path = TABLE_DIR / "spatial_holdout_feature_columns.csv"
    pd.DataFrame(feature_column_rows).to_csv(feature_columns_path, index=False)
    print(f"Saved holdout feature column table to: {feature_columns_path}")

    # ------------------------------------------------------------------
    # 4. Exclude low-confidence labels
    # ------------------------------------------------------------------
    y_all = adata.obs[LABEL_KEY].astype(str)

    trainable_mask = y_all != EXCLUDE_LABEL

    y = y_all.loc[trainable_mask].copy()
    blocks = adata.obs.loc[trainable_mask, SPATIAL_BLOCK_KEY].astype(str)

    feature_sets = {
        name: X.loc[trainable_mask].copy()
        for name, X in feature_sets_all.items()
    }

    print("\nTrainable label counts:")
    print(y.value_counts())

    print("\nTrainable spatial block counts:")
    print(blocks.value_counts().sort_index())

    # ------------------------------------------------------------------
    # 5. Run spatial holdout validation
    # ------------------------------------------------------------------
    metrics_df = run_spatial_holdout_validation(
        feature_sets=feature_sets,
        y=y,
        blocks=blocks,
    )

    print("\nSpatial holdout validation metrics:")
    print(metrics_df)

    # ------------------------------------------------------------------
    # 6. Save summary tables and figures
    # ------------------------------------------------------------------
    save_holdout_summary_tables(metrics_df)
    plot_holdout_summary_figures(metrics_df)

    print("\nMilestone 6 complete.")


if __name__ == "__main__":
    main()