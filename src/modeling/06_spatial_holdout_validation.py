"""
Milestone 4.5: Spatial holdout validation for the baseline niche classifier.

Purpose
-------
This script evaluates whether the baseline spatial niche classifier generalizes
to unseen tissue regions.

The previous ML script used a random train/test split. That is useful as a
baseline, but random spot-level splitting can overestimate performance in
spatial transcriptomics because nearby spots are spatially correlated.

This script performs a more stringent validation:

    1. Divide the tissue into spatial blocks.
    2. Hold out one full spatial block at a time.
    3. Train on the remaining blocks.
    4. Test on the held-out region.
    5. Repeat for every valid block.

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

Spatial holdout validation asks a harder and more realistic question:

    Can the model classify biological niches in a region of tissue it did not
    train on?

Input
-----
Final labeled AnnData object:

    data/processed/visium_human_breast_cancer_final_labeled.h5ad

Expected important fields:

    adata.obs["ml_training_label"]
    adata.obsm["X_pca"]
    adata.obsm["spatial"]
    adata.raw

Outputs
-------
Tables:

    results/tables/spatial_block_label_counts.csv
    results/tables/spatial_holdout_validation_metrics.csv
    results/tables/spatial_holdout_validation_summary_by_feature_set.csv
    results/tables/spatial_holdout_*_classification_report.csv

Figures:

    results/figures/06_spatial_holdout_validation/

Main analysis design
--------------------
The script compares two feature sets:

    1. with_spatial_coordinates
       PCA + QC + spatial_x/spatial_y + marker scores

    2. without_spatial_coordinates
       PCA + QC + marker scores

Why compare with and without spatial coordinates?
-------------------------------------------------
Spatial coordinates can help within-sample prediction because tissue regions
are spatially organized. However, they can also encourage the model to memorize
where each niche occurs in this one tissue section.

If the model performs similarly without spatial coordinates, that suggests it
is learning expression/marker patterns rather than simply memorizing position.

Alternative validation designs
------------------------------
Future versions could use:

    - 4 x 4 or 5 x 5 spatial grids
    - vertical or horizontal tissue strips
    - k-means clusters on spatial coordinates
    - leave-one-cluster-region-out validation
    - spatial buffering between train and test regions
    - cross-section validation using another Visium section
    - cross-patient validation using additional breast cancer samples

Alternative model choices
-------------------------
This script uses a random forest to match the strongest baseline model from
Milestone 4. Later versions could test:

    - logistic regression
    - gradient boosting
    - XGBoost / LightGBM
    - calibrated random forests
    - graph neural networks
    - spatially aware models
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This script lives in:
#   src/modeling/06_spatial_holdout_validation.py
#
# parents[2] moves from:
#   src/modeling/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Final labeled object created by:
#   src/analysis/04_apply_manual_annotations.py
#
# This object has the manually curated labels used as weak supervision.
INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_final_labeled.h5ad"
)

# Output directories for holdout figures and tables.
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "06_spatial_holdout_validation"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset and label metadata
# ---------------------------------------------------------------------
# This should match the key in adata.uns["spatial"].
LIBRARY_ID = "Visium_Human_Breast_Cancer"

# Target label column created in the manual annotation step.
LABEL_KEY = "ml_training_label"

# Low-confidence/review spots are excluded from training and testing because
# they were intentionally marked as uncertain or mixed.
EXCLUDE_LABEL = "Exclude_low_confidence"

# Name of the new spatial block column that will be added to adata.obs.
SPATIAL_BLOCK_KEY = "spatial_block_3x3"


# ---------------------------------------------------------------------
# Marker sets for feature engineering
# ---------------------------------------------------------------------
# These marker sets are scored again here so the holdout validation uses the
# same type of interpretable biological features as the baseline ML script.
#
# Why recompute them instead of loading model features from disk?
#   Recomputing keeps this script self-contained and reproducible. If the
#   AnnData object is available, the feature table can be rebuilt without
#   relying on an intermediate CSV.
#
# Future alternatives:
#   - store feature engineering in a shared utility module
#   - save and reload a full feature matrix
#   - add pathway scores
#   - add spatial-neighborhood averages
#   - add image-derived features from H&E patches
MARKER_SETS = {
    "tumor_epithelial": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "TACSTD2",
    ],
    "stromal_caf": [
        "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "ACTA2", "TAGLN",
    ],
    "pan_immune": [
        "PTPRC", "LCP1", "CD52", "CORO1A", "CXCL13",
    ],
    "t_cell": [
        "CD3D", "CD3E", "CD2", "TRAC", "CD8A", "CD8B", "GZMB", "NKG7",
    ],
    "b_cell_plasma": [
        "MS4A1", "CD79A", "CD79B", "BANK1", "MZB1", "JCHAIN",
        "IGKC", "IGHG1", "IGHG3",
    ],
    "myeloid_apc": [
        "CD74", "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "C1QA", "C1QB",
        "LYZ", "LST1",
    ],
    "endothelial": [
        "PECAM1", "VWF", "KDR", "ENG", "PLVAP",
    ],
    "proliferation": [
        "MKI67", "TOP2A", "PCNA", "MCM5", "UBE2C",
    ],
    "adipocyte_fat": [
        "FABP4", "PLIN1", "ADIPOQ", "LPL", "G0S2", "CFD",
    ],
    "hypoxia_glycolysis": [
        "GAPDH", "PGK1", "TPI1", "ENO1", "LDHA", "VEGFA", "CA9",
    ],
    "luminal_secretory": [
        "SCGB2A2", "SCGB1D2", "CSTA", "S100G", "GATA3", "XBP1",
    ],
}


def safe_name(text: str) -> str:
    """
    Convert arbitrary text into a filename-safe string.

    Why this is needed
    ------------------
    Feature set names and block IDs are used in output filenames. This helper
    prevents spaces, slashes, punctuation, or other special characters from
    producing invalid or messy filenames.

    Example
    -------
    "with spatial coordinates" -> "with_spatial_coordinates"
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
    and metric summaries. A single helper keeps figure export settings
    consistent.

    Alternative parameters:
        dpi=150  for smaller files
        dpi=300  for GitHub/presentation-quality figures
        dpi=600  for print-quality figures, but larger files
    """
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def get_available_genes(adata, genes: list[str]) -> list[str]:
    """
    Return marker genes that are present in the AnnData object.

    Why this function exists
    ------------------------
    Some marker genes may not be present after preprocessing, gene-name
    handling, or highly variable gene subsetting. Filtering to available genes
    prevents marker scoring from failing.

    Why use adata.raw?
    ------------------
    The processed object is subset to highly variable genes, but marker genes
    are not guaranteed to be highly variable. `adata.raw` preserves the full
    normalized/log-transformed expression matrix before HVG subsetting.
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
    The spatial holdout validation should evaluate the same general feature
    logic as the baseline ML model. Marker signature scores make the model more
    interpretable than PCA coordinates alone.

    Why require at least two genes?
    -------------------------------
    A single-gene score can be noisy and overly dependent on one marker.
    Requiring at least two available genes makes the score more stable.

    Alternative feature choices
    ---------------------------
    - raw curated marker expression
    - pathway scores from MSigDB/Hallmark gene sets
    - AUCell/ssGSEA/GSVA/decoupler scores
    - cell-type deconvolution proportions
    - neighborhood-smoothed marker scores
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

    Why this is included
    --------------------
    Spatial blocks define the held-out tissue regions. Holding out one block at
    a time creates a stricter test than random spot-level splitting.

    Current choice
    --------------
    n_bins_x = 3
    n_bins_y = 3

    This creates a 3 x 3 grid, or 9 possible held-out regions.

    Why 3 x 3?
    ----------
    It is a practical balance:
        - enough blocks to evaluate multiple tissue regions
        - each block is large enough to contain enough test spots
        - output plots remain easy to interpret

    Alternative parameters
    ----------------------
    n_bins_x/n_bins_y:
        - 2 x 2: larger blocks, fewer validation folds
        - 4 x 4: smaller blocks, more folds, possible sparse labels
        - 5 x 5: more detailed, but many blocks may lack enough labels

    Alternative block definitions:
        - horizontal tissue strips
        - vertical tissue strips
        - k-means clusters on spatial coordinates
        - connected components from tissue morphology
        - pathologist-defined regions
        - spatial buffering around test blocks
    """
    spatial = adata.obsm["spatial"]

    x = spatial[:, 0]
    y = spatial[:, 1]

    # pd.cut creates approximately equal-width bins along each coordinate.
    # include_lowest=True ensures boundary values are not dropped.
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


def build_feature_table(
    adata,
    score_columns: list[str],
    n_pcs: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    Build two ML feature sets for spatial holdout validation.

    Feature sets
    ------------
    1. with_spatial_coordinates:
        PCA + QC + spatial_x/spatial_y + marker scores

    2. without_spatial_coordinates:
        PCA + QC + marker scores

    Why compare these?
    ------------------
    Spatial coordinates can improve within-sample prediction, but they can also
    encourage position memorization. Comparing both feature sets helps test
    whether the model depends heavily on raw tissue coordinates.

    Why include PCA?
    ----------------
    PCA captures broad expression variation without training directly on
    thousands of genes. This is useful for a baseline model with relatively few
    spots.

    Why include QC metrics?
    -----------------------
    QC metrics may capture technical variation that affects expression-derived
    features. Including them helps the model account for technical structure,
    but feature importance should be inspected to ensure the model is not
    dominated by QC artifacts.

    Alternative parameters
    ----------------------
    n_pcs:
        - 10 or 20: less complex, possibly less overfitting
        - 30: current balanced choice
        - 50: more expression variation, possible overfitting

    Alternative feature sets:
        - marker scores only
        - PCA only
        - PCA + marker scores only
        - no QC metrics
        - spatial-neighborhood features
        - H&E image features
    """
    if "X_pca" not in adata.obsm:
        raise ValueError("Expected PCA coordinates in adata.obsm['X_pca'].")

    pcs = adata.obsm["X_pca"][:, :n_pcs]
    pc_cols = [f"PC{i + 1}" for i in range(pcs.shape[1])]

    pc_df = pd.DataFrame(
        pcs,
        index=adata.obs_names,
        columns=pc_cols,
    )

    qc_cols = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
    ]

    missing_qc = [col for col in qc_cols if col not in adata.obs.columns]
    if missing_qc:
        raise ValueError(f"Missing QC columns: {missing_qc}")

    qc_df = adata.obs[qc_cols].copy()

    spatial = adata.obsm["spatial"]
    spatial_df = pd.DataFrame(
        spatial,
        index=adata.obs_names,
        columns=["spatial_x", "spatial_y"],
    )

    signature_df = adata.obs[score_columns].copy()

    # Feature set 1 intentionally includes spatial coordinates to test the
    # strongest within-sample predictive setup.
    features_with_spatial = pd.concat(
        [
            pc_df,
            qc_df,
            spatial_df,
            signature_df,
        ],
        axis=1,
    )

    # Feature set 2 removes spatial coordinates to test whether expression and
    # marker features alone can generalize across held-out tissue regions.
    features_without_spatial = pd.concat(
        [
            pc_df,
            qc_df,
            signature_df,
        ],
        axis=1,
    )

    feature_sets = {
        "with_spatial_coordinates": features_with_spatial,
        "without_spatial_coordinates": features_without_spatial,
    }

    cleaned_feature_sets = {}

    for name, features in feature_sets.items():
        # Replace infinite values and impute missing values defensively.
        #
        # Median imputation is used because it is robust to outliers.
        #
        # Alternative:
        #   Use scikit-learn SimpleImputer in a Pipeline if this becomes a
        #   productionized workflow.
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median(numeric_only=True))
        cleaned_feature_sets[name] = features

    return cleaned_feature_sets


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    filename: str,
) -> None:
    """
    Plot a confusion matrix for one spatial holdout fold.

    Why this plot is included
    -------------------------
    Holdout metrics summarize performance, but confusion matrices show which
    biological niches are mistaken for each other in each held-out region.

    Why this matters biologically
    -----------------------------
    Misclassifying related tumor epithelial states may be less surprising than
    confusing adipocyte/fat regions with immune regions. The confusion matrix
    helps interpret errors in biological context.

    Alternative visualization options
    ---------------------------------
    - normalized confusion matrix by true label
    - normalized confusion matrix by predicted label
    - per-class recall bar plots
    - aggregate confusion matrix across all spatial folds
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


def plot_spatial_blocks(adata) -> None:
    """
    Plot spatial block assignments and ML training labels.

    Why this plot is included
    -------------------------
    Before trusting spatial holdout metrics, it is important to see what the
    held-out blocks look like on the tissue.

    This figure helps answer:
        - Are blocks covering meaningful tissue areas?
        - Are labels unevenly distributed across blocks?
        - Are some niches localized to only part of the tissue?

    Alternative:
        Plot each held-out block as train/test masks to visualize exactly what
        the model sees in each fold.
    """
    sc.pl.spatial(
        adata,
        color=[SPATIAL_BLOCK_KEY, LABEL_KEY],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_blocks_and_training_labels.png")


def summarize_block_labels(adata) -> None:
    """
    Save a table of label counts per spatial block.

    Why this is important
    ---------------------
    Spatial holdout performance depends strongly on which labels appear in each
    held-out block. Some regions may contain only a few niche classes, while
    others may contain many.

    This table helps interpret:
        - why some blocks have lower macro F1
        - whether rare classes appear in the test block
        - whether a block is too label-imbalanced for reliable evaluation

    Alternative:
        Add percentages per block to make composition easier to compare.
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


def train_and_evaluate_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    blocks: pd.Series,
    heldout_block: str,
    feature_set_name: str,
):
    """
    Train on all blocks except one, then test on the held-out block.

    Why this function exists
    ------------------------
    This is the core spatial holdout validation unit. It evaluates whether the
    model can classify spots in one spatial region after training on all other
    regions.

    Skip criteria
    -------------
    A block is skipped if:
        - it has fewer than 30 test spots
        - it contains fewer than 2 labels
        - the training set contains fewer than 2 labels

    Why skip small/low-label blocks?
    --------------------------------
    Metrics from extremely small or single-class test blocks are unstable and
    not very informative for multiclass classification.

    Alternative thresholds
    ----------------------
    Minimum test spots:
        - 20: more permissive
        - 30: current choice
        - 50: more conservative

    Minimum labels:
        - require 2 labels: current minimum for multiclass relevance
        - require 3+ labels for stricter evaluation

    Model parameters
    ----------------
    The random forest matches the baseline model from Milestone 4.

    Alternatives:
        - n_estimators=100 for faster runs
        - n_estimators=1000 for more stable forests
        - min_samples_leaf=1 for more flexible trees
        - min_samples_leaf=5 or 10 for smoother trees
        - max_depth to explicitly limit tree complexity
    """
    test_mask = blocks == heldout_block
    train_mask = ~test_mask

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    # Skip blocks that are too small for meaningful evaluation.
    if X_test.shape[0] < 30:
        print(f"  Skipping {heldout_block}: fewer than 30 test spots.")
        return None

    # A single-label test block cannot provide meaningful multiclass metrics.
    if y_test.nunique() < 2:
        print(f"  Skipping {heldout_block}: fewer than 2 labels in test block.")
        return None

    # Training requires at least two classes.
    if y_train.nunique() < 2:
        print(f"  Skipping {heldout_block}: fewer than 2 labels in training set.")
        return None

    train_labels = set(y_train.unique())
    test_labels = set(y_test.unique())

    # This is mostly a safety check. If a test label is absent from training,
    # no supervised classifier can learn that class in this fold.
    labels_missing_from_train = sorted(test_labels - train_labels)

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=3,
    )

    model.fit(X_train, y_train)

    y_pred = pd.Series(
        model.predict(X_test),
        index=y_test.index,
        name="predicted_label",
    )

    # Include labels that appear in either truth or prediction.
    # This avoids dropping predicted-only labels from the confusion matrix.
    present_labels = sorted(set(y_test.unique()).union(set(y_pred.unique())))

    accuracy = accuracy_score(y_test, y_pred)

    # Balanced accuracy is useful here because held-out blocks can be label
    # imbalanced.
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Macro F1 treats each class equally and is important when rare classes
    # exist in a held-out block.
    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )

    # Weighted F1 reflects overall performance while accounting for class sizes.
    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=present_labels,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "feature_set": feature_set_name,
        "heldout_block": heldout_block,
        "n_train_spots": X_train.shape[0],
        "n_test_spots": X_test.shape[0],
        "n_train_labels": y_train.nunique(),
        "n_test_labels": y_test.nunique(),
        "test_labels": ";".join(sorted(y_test.unique())),
        "labels_missing_from_train": ";".join(labels_missing_from_train),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    print("  Metrics:")
    print(f"    test spots:        {X_test.shape[0]}")
    print(f"    test labels:       {y_test.nunique()}")
    print(f"    accuracy:          {accuracy:.4f}")
    print(f"    balanced_accuracy: {balanced_accuracy:.4f}")
    print(f"    macro_f1:          {macro_f1:.4f}")
    print(f"    weighted_f1:       {weighted_f1:.4f}")

    if labels_missing_from_train:
        print(f"    labels missing from train: {labels_missing_from_train}")

    # Save a classification report for this specific fold.
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
        / f"spatial_holdout_{safe_name(feature_set_name)}_{safe_name(heldout_block)}_classification_report.csv"
    )
    report_df.to_csv(report_path)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=present_labels,
    )

    plot_confusion_matrix(
        cm,
        labels=present_labels,
        title=f"{feature_set_name}: held out {heldout_block}",
        filename=(
            f"spatial_holdout_{safe_name(feature_set_name)}_"
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
    Run leave-one-spatial-block-out validation for each feature set.

    Why this function exists
    ------------------------
    It applies the same spatial holdout procedure to multiple feature sets so
    their performance can be compared fairly.

    Current comparison
    ------------------
    - with_spatial_coordinates
    - without_spatial_coordinates

    Alternative comparisons
    -----------------------
    Future runs could compare:
        - PCA only
        - marker scores only
        - PCA + marker scores
        - PCA + marker scores + QC
        - H&E image features
        - spatial-neighborhood features
    """
    all_metrics = []

    unique_blocks = sorted(blocks.unique())

    for feature_set_name, X in feature_sets.items():
        print(f"\n==============================")
        print(f"Feature set: {feature_set_name}")
        print(f"Number of features: {X.shape[1]}")
        print(f"==============================")

        for heldout_block in unique_blocks:
            print(f"\nRunning spatial holdout for block: {heldout_block}")

            metrics = train_and_evaluate_holdout(
                X=X,
                y=y,
                blocks=blocks,
                heldout_block=heldout_block,
                feature_set_name=feature_set_name,
            )

            if metrics is not None:
                all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = TABLE_DIR / "spatial_holdout_validation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved spatial holdout metrics to: {metrics_path}")

    return metrics_df


def plot_holdout_metrics(metrics_df: pd.DataFrame) -> None:
    """
    Plot spatial holdout performance across blocks and feature sets.

    Why these plots are included
    ----------------------------
    The main scientific question is not only whether the model performs well
    overall, but whether performance is consistent across tissue regions.

    A block-level performance plot makes it easy to see:
        - which tissue regions are hardest
        - whether spatial coordinates help or hurt
        - whether performance varies more than random split metrics suggest

    Metrics plotted
    ---------------
    macro_f1:
        Best for understanding performance across imbalanced classes.

    balanced_accuracy:
        Useful when label composition differs across blocks.

    accuracy:
        Easy to interpret, but can be inflated by large classes.

    Alternative summary options
    ---------------------------
    - boxplots by feature set
    - mean +/- standard deviation bar plots
    - per-class recall across blocks
    - aggregate confusion matrix across all holdouts
    """
    if metrics_df.empty:
        print("No holdout metrics available to plot.")
        return

    for metric in ["macro_f1", "balanced_accuracy", "accuracy"]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for feature_set, sub_df in metrics_df.groupby("feature_set"):
            sub_df = sub_df.sort_values("heldout_block")
            ax.plot(
                sub_df["heldout_block"],
                sub_df[metric],
                marker="o",
                label=feature_set,
            )

        ax.set_xlabel("Held-out spatial block")
        ax.set_ylabel(metric)
        ax.set_title(f"Spatial holdout validation: {metric}")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.xticks(rotation=45, ha="right")

        save_current_fig(f"spatial_holdout_{metric}_by_block_feature_sets.png")

    # Save summary statistics by feature set.
    # This gives a compact table for README reporting.
    summary = (
        metrics_df.groupby("feature_set")[
            ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
        ]
        .agg(["mean", "std", "min", "max"])
    )

    summary_path = TABLE_DIR / "spatial_holdout_validation_summary_by_feature_set.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary metrics to: {summary_path}")

    print("\nSpatial holdout summary by feature set:")
    print(summary)


def main() -> None:
    """
    Run spatial holdout validation.

    Workflow
    --------
    1. Load final labeled AnnData object.
    2. Add marker signature scores.
    3. Divide tissue into spatial blocks.
    4. Summarize label composition by block.
    5. Build feature sets with and without spatial coordinates.
    6. Exclude low-confidence labels.
    7. Run leave-one-block-out validation.
    8. Save metrics, classification reports, and plots.
    """
    print(f"Loading final labeled AnnData object from: {INPUT_H5AD}")

    # Load the final manually labeled object.
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded object:")
    print(adata)

    # Fail early if the label column is missing.
    if LABEL_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find label column: {LABEL_KEY}")

    print("\nOriginal ML label counts:")
    print(adata.obs[LABEL_KEY].astype(str).value_counts())

    # Recompute marker score features so this script can run independently.
    adata, score_columns = add_marker_signature_scores(adata)

    # Add spatial blocks for leave-one-region-out validation.
    #
    # Current setting:
    #   3 x 3 grid
    #
    # Alternative:
    #   Try 4 x 4 later to test smaller held-out regions.
    adata = add_spatial_blocks(adata, n_bins_x=3, n_bins_y=3)

    print("\nSpatial block counts:")
    print(adata.obs[SPATIAL_BLOCK_KEY].astype(str).value_counts().sort_index())

    # Save a table and figure to help interpret fold difficulty.
    summarize_block_labels(adata)
    plot_spatial_blocks(adata)

    # Build feature sets for every spot.
    feature_sets_all = build_feature_table(
        adata,
        score_columns=score_columns,
        n_pcs=30,
    )

    y_all = adata.obs[LABEL_KEY].astype(str)

    # Exclude low-confidence labels from both training and testing. This keeps
    # the holdout validation focused on biologically defensible labels.
    trainable_mask = y_all != EXCLUDE_LABEL

    y = y_all.loc[trainable_mask].copy()
    blocks = adata.obs.loc[trainable_mask, SPATIAL_BLOCK_KEY].astype(str)

    feature_sets = {
        name: X.loc[trainable_mask].copy()
        for name, X in feature_sets_all.items()
    }

    print("\nTrainable label counts:")
    print(y.value_counts())

    metrics_df = run_spatial_holdout_validation(
        feature_sets=feature_sets,
        y=y,
        blocks=blocks,
    )

    print("\nSpatial holdout validation metrics:")
    print(metrics_df)

    plot_holdout_metrics(metrics_df)

    print("\nMilestone 4.5 complete.")


if __name__ == "__main__":
    main()
