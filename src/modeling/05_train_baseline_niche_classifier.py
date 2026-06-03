"""
Milestone 5: Train baseline ML models for spatial niche classification.

Purpose
-------
This script trains the first supervised machine learning models for the
SpatialNicheAI project.

The goal is to learn a mapping from interpretable spot-level features to the
manual biological niche labels created in the previous annotation step.

Conceptually, the model learns:

    expression-derived features + QC metrics + spatial coordinates + marker scores
        -> manual biological niche label

Why this script exists
----------------------
The earlier analysis steps identify clusters and manually annotate those
clusters using marker genes and spatial patterns. This script tests whether
those manually interpreted biological niches can be predicted from quantitative
features.

This is intentionally a baseline modeling script. The goal is not to claim a
final clinical model. The goal is to establish:

    1. A reproducible ML training workflow.
    2. A clear feature engineering strategy.
    3. A first model comparison.
    4. A saved model artifact.
    5. Spatial prediction and confidence maps.
    6. A foundation for more rigorous validation later.

Input
-----
Final labeled AnnData object:

    data/processed/visium_human_breast_cancer_final_labeled.h5ad

Expected important columns:
    adata.obs["ml_training_label"]
    adata.obs["manual_niche_label_short"]
    adata.obs["manual_label_confidence"]

Expected embeddings/coordinates:
    adata.obsm["X_pca"]
    adata.obsm["spatial"]

Output
------
Prediction-annotated AnnData object:

    data/processed/visium_human_breast_cancer_ml_predictions.h5ad

Model artifacts:

    models/baseline_spatial_niche_classifier.joblib
    models/baseline_label_encoder.joblib
    models/baseline_model_metadata.json

Tables:

    results/tables/baseline_ml_model_metrics.csv
    results/tables/logistic_regression_classification_report.csv
    results/tables/random_forest_classification_report.csv
    results/tables/baseline_random_forest_feature_importance.csv
    results/tables/baseline_ml_spot_predictions.csv

Figures:

    results/figures/05_baseline_ml/

Important scientific caution
----------------------------
The labels used here are weakly supervised labels derived from manual
annotation of Leiden clusters. A random spot-level train/test split can
overestimate performance in spatial transcriptomics because neighboring spots
are correlated and labels are partly cluster-derived.

That is why the next script performs spatial holdout validation:

    src/modeling/06_spatial_holdout_validation.py

Alternative modeling options
----------------------------
Future models could include:

    - XGBoost / LightGBM
    - support vector machines
    - calibrated random forests
    - elastic net multinomial logistic regression
    - graph-based features from spatial neighbors
    - graph neural networks
    - H&E image patch features
    - cross-sample validation using additional breast cancer Visium datasets

Alternative feature options
---------------------------
Future feature sets could include:

    - more or fewer PCA components
    - raw marker gene expression values
    - pathway enrichment scores
    - spatial-neighborhood averaged expression
    - local spatial diversity features
    - cell-type deconvolution proportions
    - inferred CNV/tumor scores
    - H&E image embeddings
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This script lives in:
#   src/modeling/05_train_baseline_niche_classifier.py
#
# parents[2] moves from:
#   src/modeling/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Final labeled object created by:
#   src/analysis/04_apply_manual_annotations.py
#
# This object contains the manually curated labels used for supervised ML.
INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_final_labeled.h5ad"
)

# Output object with model predictions added to adata.obs.
OUTPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_ml_predictions.h5ad"
)

# Output directories.
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "05_baseline_ml"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
MODEL_DIR = PROJECT_ROOT / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset and label metadata
# ---------------------------------------------------------------------
# This should match the key in adata.uns["spatial"].
LIBRARY_ID = "Visium_Human_Breast_Cancer"

# Target label column created in the manual annotation step.
# This column keeps high-confidence biological labels and groups uncertain
# clusters under EXCLUDE_LABEL.
LABEL_KEY = "ml_training_label"

# Spots with this label are excluded from supervised model training.
# They are still predicted after training so we can inspect what the model
# would assign to uncertain/review regions.
EXCLUDE_LABEL = "Exclude_low_confidence"


# ---------------------------------------------------------------------
# Marker sets for feature engineering
# ---------------------------------------------------------------------
# These marker sets are converted into per-spot signature scores and used as
# interpretable ML features.
#
# Why use marker scores as features?
#   PCA features capture global expression variation, but marker scores encode
#   biologically interpretable programs. This makes the model more explainable
#   and lets feature importance highlight meaningful signatures.
#
# Important:
#   Marker scores are not pure cell-type proportions. Visium spots can contain
#   mixtures of cell types, and scores should be interpreted as relative
#   program activity.
#
# Future alternatives:
#   - Use pathway scores instead of marker scores.
#   - Add spatial-neighborhood averaged marker scores.
#   - Use deconvolution-derived cell-type proportions.
#   - Use raw expression for a curated marker panel.
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


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.

    Why this helper exists
    ----------------------
    This script generates many figures. Centralizing save behavior ensures all
    plots use the same DPI, output directory, and closing behavior.

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
    Some genes may be missing because of gene filtering, gene symbol handling,
    or HVG subsetting. Using only available genes prevents Scanpy from failing
    when scoring marker sets.

    Why use adata.raw?
    ------------------
    The processed AnnData object was subset to highly variable genes for PCA
    and clustering. Marker genes are not guaranteed to be HVGs. `adata.raw`
    preserves the full normalized/log-transformed expression matrix before HVG
    subsetting, so it is preferred for marker scoring.

    Alternative:
        If using Ensembl IDs, add a symbol-to-Ensembl mapping step before
        checking gene availability.
    """
    if adata.raw is not None:
        available = set(adata.raw.var_names)
    else:
        available = set(adata.var_names)

    return [gene for gene in genes if gene in available]


def add_marker_signature_scores(adata):
    """
    Add marker signature scores to adata.obs.

    Why this step is included
    -------------------------
    Marker signature scores provide interpretable biological features for the
    ML model. They complement PCA features, which are useful but less directly
    interpretable.

    The resulting columns have names like:
        score_tumor_epithelial
        score_myeloid_apc
        score_adipocyte_fat

    How Scanpy scoring works
    ------------------------
    `sc.tl.score_genes` compares the expression of a marker gene list against
    a set of control genes with similar expression levels. The score is useful
    as a relative enrichment-like signal.

    Alternative feature approaches
    ------------------------------
    - Use raw expression of marker genes directly.
    - Use pathway scores from MSigDB/Hallmark gene sets.
    - Use AUCell, ssGSEA, GSVA, or decoupler scores.
    - Use deconvolution proportions from a breast cancer scRNA-seq reference.
    - Compute spatial-neighborhood averaged marker scores.
    """
    score_columns = []

    for signature_name, genes in MARKER_SETS.items():
        available_genes = get_available_genes(adata, genes)

        print(f"\nSignature: {signature_name}")
        print(f"  Available genes: {available_genes}")

        # Require at least two genes so that a score is not driven by a single
        # marker. For highly specific rare markers, a one-gene feature could be
        # useful, but it should be interpreted cautiously.
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


def build_feature_table(
    adata,
    score_columns: list[str],
    n_pcs: int = 30,
) -> pd.DataFrame:
    """
    Build the feature matrix used for supervised ML.

    Feature groups
    --------------
    1. PCA coordinates
    2. QC metrics
    3. spatial coordinates
    4. marker signature scores

    Why these features are included
    -------------------------------
    PCA coordinates:
        Capture broad gene expression variation while reducing dimensionality.
        This avoids training directly on tens of thousands of genes for a small
        number of spots.

    QC metrics:
        Help the model account for technical variation such as library size or
        detected gene count. Including QC features can be informative, but they
        should be interpreted cautiously because a model could learn technical
        artifacts.

    Spatial coordinates:
        Provide positional information. This can help predict tissue regions,
        but it can also encourage memorization of this specific tissue section.
        The spatial holdout script later tests robustness with and without
        spatial coordinates.

    Marker signature scores:
        Provide interpretable biological signal that aligns with the manual
        annotation process.

    Important limitation
    --------------------
    This feature set is still single-sample. A model trained this way may not
    generalize to another patient or tissue section without external validation.

    Alternative parameters
    ----------------------
    n_pcs:
        - 10 or 20: simpler model, less noise
        - 30: current balanced choice
        - 50: more variation, possible overfitting

    Alternative feature sets:
        - omit spatial_x/spatial_y
        - add spatial-neighborhood averages
        - add graph connectivity features
        - add raw curated marker expression
        - add pathway scores
        - add H&E image features
    """
    if "X_pca" not in adata.obsm:
        raise ValueError("Expected PCA coordinates in adata.obsm['X_pca'].")

    # Use the first n_pcs principal components.
    # These are compact expression-derived features.
    pcs = adata.obsm["X_pca"][:, :n_pcs]
    pc_cols = [f"PC{i + 1}" for i in range(pcs.shape[1])]
    pc_df = pd.DataFrame(pcs, index=adata.obs_names, columns=pc_cols)

    # QC features are included because library size and detection rate can
    # influence expression-derived features. They also help diagnose whether
    # the model is relying heavily on technical variation.
    qc_cols = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
    ]

    missing_qc = [col for col in qc_cols if col not in adata.obs.columns]
    if missing_qc:
        raise ValueError(f"Missing QC columns: {missing_qc}")

    qc_df = adata.obs[qc_cols].copy()

    # Spatial coordinates describe each spot's physical location in the tissue
    # image coordinate system.
    #
    # Note:
    #   These features are useful for within-sample prediction, but they can
    #   inflate performance if the model memorizes tissue position. That is why
    #   the next script compares holdout performance with and without them.
    spatial = adata.obsm["spatial"]
    spatial_df = pd.DataFrame(
        spatial,
        index=adata.obs_names,
        columns=["spatial_x", "spatial_y"],
    )

    # Marker scores created by add_marker_signature_scores.
    signature_df = adata.obs[score_columns].copy()

    # Concatenate all feature groups into one model-ready table.
    features = pd.concat(
        [
            pc_df,
            qc_df,
            spatial_df,
            signature_df,
        ],
        axis=1,
    )

    # Clean unexpected numerical issues.
    #
    # Why median imputation?
    #   For numeric features, the median is robust to outliers and avoids
    #   dropping spots if a rare missing value appears.
    #
    # Alternatives:
    #   - mean imputation
    #   - KNN imputation
    #   - dropping rows with missing values
    #   - fitting a scikit-learn SimpleImputer inside a Pipeline
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))

    return features


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    filename: str,
) -> None:
    """
    Plot a confusion matrix.

    Why this plot is included
    -------------------------
    Overall metrics can hide class-specific mistakes. A confusion matrix shows
    which biological niches are confused with each other.

    For this project, confusion patterns are biologically meaningful. For
    example, confusing two epithelial-like tumor states may be less concerning
    than confusing immune and tumor regions.

    Alternative visualization options
    ---------------------------------
    - normalized confusion matrix by true label
    - normalized confusion matrix by predicted label
    - seaborn heatmap with annotations
    - per-class precision/recall bar plots
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add raw counts to each confusion matrix cell.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    save_current_fig(filename)


def plot_feature_importance(
    model,
    feature_names: list[str],
    filename: str,
    top_n: int = 30,
) -> None:
    """
    Plot random forest feature importance.

    Why this is included
    --------------------
    Feature importance helps interpret what the random forest used to separate
    niche labels. This is important because the project is meant to be
    biologically interpretable, not just predictive.

    Important limitation
    --------------------
    Random forest impurity-based feature importance can be biased toward
    continuous variables or variables with many possible split points. It is a
    useful first-pass diagnostic, not definitive biological proof.

    Alternative interpretation methods
    ----------------------------------
    - permutation importance
    - SHAP values
    - logistic regression coefficients
    - one-vs-rest feature importance per class
    - ablation testing by removing feature groups
    - comparing models with and without spatial coordinates

    Parameters
    ----------
    top_n:
        Number of top features to show in the plot.
        Useful values:
            20 for a compact README figure
            30 for a balanced summary
            50 for deeper exploratory analysis
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not expose feature_importances_; skipping plot.")
        return

    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    top_importances = importances.head(top_n).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    top_importances.plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} feature importances")
    save_current_fig(filename)

    importance_path = TABLE_DIR / "baseline_random_forest_feature_importance.csv"

    # This pandas-compatible form avoids using reset_index(names=...), which
    # failed in one earlier environment.
    importance_df = (
        importances
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importances to: {importance_path}")


def evaluate_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
):
    """
    Train and evaluate a supervised classifier.

    Why this function exists
    ------------------------
    Both baseline models should be evaluated in the same way. This helper
    keeps model comparison consistent.

    Metrics
    -------
    accuracy:
        Fraction of correctly classified spots. Easy to understand, but can be
        misleading with class imbalance.

    balanced_accuracy:
        Average recall across classes. More useful when some niche classes are
        much smaller than others.

    macro_f1:
        F1 score averaged equally across classes. This is the main model
        selection metric because it prevents large classes from dominating the
        comparison.

    weighted_f1:
        F1 score weighted by class frequency. Useful for overall performance,
        but can hide poor performance on rare classes.

    Alternative evaluation options
    ------------------------------
    - cross-validation
    - spatial holdout validation
    - per-class ROC-AUC in one-vs-rest form
    - top-k accuracy
    - calibration curves
    - precision-recall curves for rare classes
    """
    print(f"\nTraining model: {model_name}")

    # Fit the model using only high-confidence labeled spots.
    model.fit(X_train, y_train)

    # Predict labels for the held-out random test set.
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
    }

    print(f"\n{model_name} metrics:")
    for key, value in metrics.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")

    # Save a detailed per-class report.
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_path = TABLE_DIR / f"{model_name}_classification_report.csv"
    report_df.to_csv(report_path)
    print(f"Saved classification report to: {report_path}")

    # Save confusion matrix plot.
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm,
        labels=label_names,
        title=f"{model_name} confusion matrix",
        filename=f"{model_name}_confusion_matrix.png",
    )

    return metrics, model


def main() -> None:
    """
    Train baseline models and generate predictions.

    Workflow
    --------
    1. Load final labeled AnnData object.
    2. Add marker signature scores.
    3. Build ML feature matrix.
    4. Exclude low-confidence labels from training.
    5. Encode labels for scikit-learn.
    6. Create a stratified train/test split.
    7. Train logistic regression and random forest models.
    8. Select the best model by macro F1.
    9. Save model artifacts and metadata.
    10. Predict labels for every spot, including excluded spots.
    11. Save prediction tables and plots.
    """
    print(f"Loading final labeled AnnData object from: {INPUT_H5AD}")

    # Load the final labeled object from Milestone 3.5.
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded object:")
    print(adata)

    # Fail early if the target label column is missing.
    if LABEL_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find label column: {LABEL_KEY}")

    print("\nOriginal ML label counts:")
    print(adata.obs[LABEL_KEY].value_counts())

    # ------------------------------------------------------------------
    # 1. Add marker signature scores
    # ------------------------------------------------------------------
    # These scores are used as interpretable biological features.
    adata, score_columns = add_marker_signature_scores(adata)

    print("\nMarker signature score columns:")
    print(score_columns)

    # ------------------------------------------------------------------
    # 2. Build feature matrix
    # ------------------------------------------------------------------
    # X_all contains features for every spot, including low-confidence spots.
    # We train on high-confidence spots only, but later predict all spots.
    X_all = build_feature_table(
        adata,
        score_columns=score_columns,
        n_pcs=30,
    )

    y_all = adata.obs[LABEL_KEY].astype(str)

    # Save a small feature preview for transparency and debugging.
    # The full feature matrix is not saved by default to avoid unnecessary
    # large intermediate files.
    feature_table_path = TABLE_DIR / "baseline_ml_feature_table_preview.csv"
    X_all.head(50).to_csv(feature_table_path)
    print(f"\nSaved feature table preview to: {feature_table_path}")

    # ------------------------------------------------------------------
    # 3. Exclude low-confidence labels for supervised training
    # ------------------------------------------------------------------
    # Low-confidence/review spots are not used to fit the model because their
    # labels were explicitly marked as uncertain or mixed.
    #
    # Alternative:
    #   Keep low-confidence spots as an "Uncertain" class, but that would train
    #   the model to reproduce uncertainty rather than clear biological niches.
    train_mask = y_all != EXCLUDE_LABEL

    X = X_all.loc[train_mask].copy()
    y = y_all.loc[train_mask].copy()

    print("\nTraining label counts after excluding low-confidence spots:")
    print(y.value_counts())

    # scikit-learn classifiers usually expect numeric labels.
    # LabelEncoder provides a stable mapping between class names and integers.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_names = label_encoder.classes_.tolist()

    # Save label mapping so model outputs can be interpreted later.
    label_mapping = pd.DataFrame(
        {
            "encoded_label": range(len(label_names)),
            "label_name": label_names,
        }
    )

    label_mapping_path = TABLE_DIR / "baseline_ml_label_mapping.csv"
    label_mapping.to_csv(label_mapping_path, index=False)
    print(f"Saved label mapping to: {label_mapping_path}")

    # ------------------------------------------------------------------
    # 4. Stratified train/test split
    # ------------------------------------------------------------------
    # Stratification preserves approximate class proportions in train and test.
    #
    # Current choice:
    #   test_size=0.25 gives a 75/25 split.
    #
    # Alternative useful parameters:
    #   test_size=0.20 for more training data
    #   test_size=0.30 for a larger evaluation set
    #   random_state values other than 42 to check stability
    #
    # Important limitation:
    #   This split is random across spots and may overestimate performance
    #   because nearby spots are spatially correlated.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    print("\nTrain/test sizes:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")

    # ------------------------------------------------------------------
    # 5. Define baseline models
    # ------------------------------------------------------------------
    # Logistic regression:
    #   A simple, interpretable linear baseline.
    #   StandardScaler is needed because logistic regression is sensitive to
    #   feature scale.
    #
    # Random forest:
    #   A nonlinear baseline that can capture interactions between PCA features,
    #   marker scores, QC metrics, and spatial coordinates.
    #
    # Why compare both?
    #   If logistic regression performs similarly to random forest, the class
    #   boundaries may be mostly linear in the engineered feature space. If
    #   random forest performs better, nonlinear interactions may help.
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # Random forest parameter notes:
    #   n_estimators=500
    #       More trees usually stabilize performance, but increase runtime.
    #       Alternatives: 100, 300, 1000.
    #
    #   class_weight="balanced"
    #       Helps account for label imbalance.
    #
    #   min_samples_leaf=3
    #       Reduces overly specific leaves and helps prevent overfitting.
    #       Alternatives: 1 for more flexible trees, 5 or 10 for smoother trees.
    #
    #   n_jobs=-1
    #       Use all available CPU cores.
    random_forest = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=3,
    )

    models = {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }

    # ------------------------------------------------------------------
    # 6. Train and evaluate
    # ------------------------------------------------------------------
    all_metrics = []
    fitted_models = {}

    for model_name, model in models.items():
        metrics, fitted_model = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_names=label_names,
        )
        all_metrics.append(metrics)
        fitted_models[model_name] = fitted_model

    # Select best model by macro F1 because classes are imbalanced and rare
    # niche classes should matter.
    metrics_df = pd.DataFrame(all_metrics).sort_values(
        "macro_f1",
        ascending=False,
    )

    metrics_path = TABLE_DIR / "baseline_ml_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved model metrics to: {metrics_path}")

    print("\nModel comparison:")
    print(metrics_df)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print(f"\nBest model by macro F1: {best_model_name}")

    # Random forest exposes feature_importances_, so plot it for interpretability.
    if "random_forest" in fitted_models:
        plot_feature_importance(
            fitted_models["random_forest"],
            feature_names=X.columns,
            filename="random_forest_feature_importance.png",
            top_n=30,
        )

    # ------------------------------------------------------------------
    # 7. Save best model and metadata
    # ------------------------------------------------------------------
    # Saving both the model and label encoder is important because numeric
    # predictions must be mapped back to label names.
    model_path = MODEL_DIR / "baseline_spatial_niche_classifier.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")

    encoder_path = MODEL_DIR / "baseline_label_encoder.joblib"
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved label encoder to: {encoder_path}")

    # Metadata makes the model artifact more reproducible.
    # It records:
    #   - which model was selected
    #   - which label column was used
    #   - which label was excluded
    #   - feature columns used during training
    #   - class names
    #   - evaluation metrics
    metadata = {
        "best_model": best_model_name,
        "label_key": LABEL_KEY,
        "excluded_label": EXCLUDE_LABEL,
        "feature_columns": X.columns.tolist(),
        "label_names": label_names,
        "metrics": metrics_df.to_dict(orient="records"),
    }

    metadata_path = MODEL_DIR / "baseline_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved model metadata to: {metadata_path}")

    # ------------------------------------------------------------------
    # 8. Predict every spot, including low-confidence/review spots
    # ------------------------------------------------------------------
    # The model is trained only on high-confidence spots, but we apply it to
    # all spots to see what label the model assigns to uncertain regions.
    #
    # Prediction confidence:
    #   For models with predict_proba, confidence is the highest predicted
    #   class probability.
    #
    # Important limitation:
    #   Random forest probabilities are not always well-calibrated.
    #
    # Alternative:
    #   Use CalibratedClassifierCV for better probability calibration.
    if hasattr(best_model, "predict_proba"):
        pred_encoded_all = best_model.predict(X_all)
        pred_proba_all = best_model.predict_proba(X_all)
        pred_confidence_all = pred_proba_all.max(axis=1)
    else:
        pred_encoded_all = best_model.predict(X_all)
        pred_confidence_all = np.full(shape=X_all.shape[0], fill_value=np.nan)

    pred_labels_all = label_encoder.inverse_transform(pred_encoded_all)

    adata.obs["baseline_ml_predicted_label"] = pd.Categorical(pred_labels_all)
    adata.obs["baseline_ml_prediction_confidence"] = pred_confidence_all

    # Save spot-level predictions for downstream review.
    prediction_table = adata.obs[
        [
            LABEL_KEY,
            "manual_niche_label_short",
            "manual_label_confidence",
            "baseline_ml_predicted_label",
            "baseline_ml_prediction_confidence",
        ]
    ].copy()

    prediction_table_path = TABLE_DIR / "baseline_ml_spot_predictions.csv"
    prediction_table.to_csv(prediction_table_path)
    print(f"Saved spot-level predictions to: {prediction_table_path}")

    # ------------------------------------------------------------------
    # 9. Plot predictions and confidence spatially
    # ------------------------------------------------------------------
    # Spatial prediction map:
    #   Shows the model's predicted niche labels across the tissue.
    sc.pl.spatial(
        adata,
        color=["baseline_ml_predicted_label"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_baseline_ml_predicted_labels.png")

    # Spatial confidence map:
    #   Highlights regions where the model is more or less confident.
    #   Low-confidence areas may correspond to mixed, transitional, or
    #   biologically ambiguous tissue regions.
    sc.pl.spatial(
        adata,
        color=["baseline_ml_prediction_confidence"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_baseline_ml_prediction_confidence.png")

    # UMAP comparison:
    #   Shows manual labels, ML predictions, and prediction confidence in
    #   expression embedding space.
    sc.pl.umap(
        adata,
        color=[
            "manual_niche_label_short",
            "baseline_ml_predicted_label",
            "baseline_ml_prediction_confidence",
        ],
        frameon=False,
        show=False,
    )
    save_current_fig("umap_manual_vs_ml_predictions.png")

    # ------------------------------------------------------------------
    # 10. Save prediction-annotated AnnData
    # ------------------------------------------------------------------
    # This output keeps the original labeled object plus model predictions and
    # prediction confidence scores.
    adata.write_h5ad(OUTPUT_H5AD)
    print(f"\nSaved prediction-annotated AnnData object to: {OUTPUT_H5AD}")

    print("\nMilestone 5 complete.")


if __name__ == "__main__":
    main()
