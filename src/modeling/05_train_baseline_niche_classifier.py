"""
Milestone 5: Train baseline ML models for spatial niche classification.

Purpose
-------
This script trains and compares five supervised machine learning models for
manual spatial niche classification.

The model learns:

    PCA expression features
    + QC metrics
    + spatial coordinates
    + marker signature scores
    + Squidpy spatial-neighborhood marker features
        -> manually curated niche label

Important caution
-----------------
This is a baseline random train/test split. In spatial transcriptomics, random
spot-level splits can overestimate performance because neighboring spots are
correlated. Spatial holdout validation is handled in script 06.

Input
-----
    data/processed/visium_human_breast_cancer_final_labeled.h5ad

Outputs
-------
    data/processed/visium_human_breast_cancer_ml_predictions.h5ad

    models/baseline_spatial_niche_classifier.joblib
    models/baseline_label_encoder.joblib
    models/baseline_model_metadata.json

    results/tables/baseline_ml_model_metrics.csv
    results/tables/baseline_ml_spot_predictions.csv
    results/tables/baseline_ml_label_mapping.csv
    results/tables/baseline_ml_feature_columns.csv

    results/figures/05_baseline_ml/

Models tested
-------------
1. Logistic regression
2. Calibrated linear SVM
3. Random forest
4. Extra trees
5. Histogram gradient boosting
"""

from pathlib import Path
import json
import re

import joblib
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_final_labeled.h5ad"
)

OUTPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_ml_predictions.h5ad"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "05_baseline_ml"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
MODEL_DIR = PROJECT_ROOT / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------
LIBRARY_ID = "Visium_Human_Breast_Cancer"

LABEL_KEY = "ml_training_label"
EXCLUDE_LABEL = "Exclude_low_confidence"

SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"


# ---------------------------------------------------------------------
# Marker sets used as interpretable ML features
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
    """Convert model names into filename-safe strings."""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_")


def save_current_fig(filename: str) -> None:
    """Save and close the active matplotlib figure."""
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
    ncols: int = 3,
) -> None:
    """
    Plot spatial data with Squidpy.

    Note
    ----
    Do not pass show=False to sq.pl.spatial_scatter in this environment because
    unsupported plotting arguments can be forwarded to Matplotlib.
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
    """Check required fields before model training."""
    required_obs = [
        LABEL_KEY,
        "manual_niche_label_short",
        "manual_label_confidence",
    ]

    missing_obs = [col for col in required_obs if col not in adata.obs.columns]
    if missing_obs:
        raise ValueError(f"Missing required adata.obs columns: {missing_obs}")

    if "X_pca" not in adata.obsm:
        raise ValueError("Missing adata.obsm['X_pca'].")

    if "spatial" not in adata.obsm:
        raise ValueError("Missing adata.obsm['spatial'].")

    if SPATIAL_CONNECTIVITY_KEY not in adata.obsp:
        print(
            f"Warning: {SPATIAL_CONNECTIVITY_KEY} not found. "
            "Neighborhood marker features will be skipped."
        )


def get_available_genes(adata, genes: list[str]) -> list[str]:
    """
    Return marker genes present in the AnnData object.

    adata.raw is preferred because the active object was subset to highly
    variable genes, while adata.raw preserves the full normalized expression
    matrix.
    """
    if adata.raw is not None:
        available = set(adata.raw.var_names)
    else:
        available = set(adata.var_names)

    return [gene for gene in genes if gene in available]


def add_marker_signature_scores(adata):
    """
    Add marker signature scores to adata.obs.

    These are interpretable biological features, such as:
        score_tumor_epithelial
        score_myeloid_apc
        score_hypoxia_glycolysis
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


def make_neighbor_score_features(
    adata,
    score_columns: list[str],
) -> pd.DataFrame:
    """
    Use the Squidpy spatial graph to calculate neighbor-averaged marker scores.

    For each marker score column:
        neighbor_mean_score_X = average score_X among spatial neighbors

    These features add local tissue context to the ML feature matrix.
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


def build_feature_table(
    adata,
    score_columns: list[str],
    n_pcs: int = 30,
) -> pd.DataFrame:
    """
    Build the supervised ML feature table.

    Feature groups:
        1. PCA coordinates
        2. QC metrics
        3. spatial coordinates
        4. marker signature scores
        5. neighbor-averaged marker signature scores
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

    features = pd.concat(
        [
            pc_df,
            qc_df,
            spatial_df,
            signature_df,
            neighbor_df,
        ],
        axis=1,
    )

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))

    return features


def define_models() -> dict:
    """
    Define five candidate classifiers.

    Models:
        1. Logistic regression
        2. Calibrated linear SVM
        3. Random forest
        4. Extra trees
        5. Histogram gradient boosting
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
    
def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    filename: str,
) -> None:
    """Plot a raw-count confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 9))
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
                fontsize=8,
            )
 
    fig.colorbar(im, ax=ax)
    save_current_fig(filename)
 
 
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
    Fit one model and save metrics, classification report, and confusion matrix.
 
    Why these metrics are used
    --------------------------
    accuracy:
        Overall fraction of correct predictions. Easy to understand, but can be
        inflated by large classes.
 
    balanced_accuracy:
        Average recall across classes. Useful for imbalanced labels.
 
    macro_f1:
        F1 score averaged equally across classes. This is the main comparison
        metric because each biological niche should matter.
 
    weighted_f1:
        F1 score weighted by class size. Useful for overall performance, but it
        can hide poor performance on rare classes.
    """
    print(f"\nTraining model: {model_name}")
 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    labels = list(range(len(label_names)))
 
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(
            f1_score(
                y_test,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }
 
    print(f"\n{model_name} metrics:")
    for key, value in metrics.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")
 
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
 
    report_df = pd.DataFrame(report).transpose()
    report_path = TABLE_DIR / f"{safe_name(model_name)}_classification_report.csv"
    report_df.to_csv(report_path)
    print(f"Saved classification report to: {report_path}")
 
    cm = confusion_matrix(y_test, y_pred, labels=labels)
 
    plot_confusion_matrix(
        cm=cm,
        labels=label_names,
        title=f"{model_name} confusion matrix",
        filename=f"{safe_name(model_name)}_confusion_matrix.png",
    )
 
    return metrics, model
 
 
def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    """Create model comparison plots."""
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
 
    plot_df = metrics_df.set_index("model")[metric_cols]
    plot_df = plot_df.sort_values("macro_f1", ascending=True)
 
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df["macro_f1"].plot(kind="barh", ax=ax)
    ax.set_xlabel("Macro F1")
    ax.set_ylabel("Model")
    ax.set_title("Model comparison by macro F1")
    ax.set_xlim(0, 1.05)
    save_current_fig("model_comparison_macro_f1.png")
 
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df[metric_cols].plot(kind="barh", ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Model")
    ax.set_title("Baseline ML model comparison")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")
    save_current_fig("model_comparison_all_metrics.png")
 
 
def plot_tree_feature_importance(
    model_name: str,
    model,
    feature_names: list[str],
    top_n: int = 30,
) -> None:
    """
    Save and plot feature importance for tree models.
 
    Important limitation
    --------------------
    Tree impurity-based feature importance is useful as a first-pass model
    interpretation tool, but it should not be treated as definitive biological
    proof. Future improvements could use permutation importance, SHAP values,
    or feature-group ablation.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not expose feature_importances_; skipping.")
        return
 
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
 
    importance_df = (
        importances
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
 
    importance_path = TABLE_DIR / f"{safe_name(model_name)}_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importances to: {importance_path}")
 
    top_importances = importances.head(top_n).sort_values(ascending=True)
 
    fig, ax = plt.subplots(figsize=(8, 8))
    top_importances.plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_name}: top {top_n} feature importances")
    save_current_fig(f"{safe_name(model_name)}_feature_importance.png")
 
 
def main() -> None:
    """Run the full five-model baseline ML workflow."""
    print(f"Loading final labeled AnnData object from: {INPUT_H5AD}")
 
    adata = sc.read_h5ad(INPUT_H5AD)
 
    print("\nLoaded object:")
    print(adata)
 
    validate_inputs(adata)
 
    print("\nOriginal ML label counts:")
    print(adata.obs[LABEL_KEY].value_counts())
 
    # ------------------------------------------------------------------
    # 1. Add marker signature scores
    # ------------------------------------------------------------------
    adata, score_columns = add_marker_signature_scores(adata)
 
    print("\nMarker signature score columns:")
    print(score_columns)
 
    # ------------------------------------------------------------------
    # 2. Build features
    # ------------------------------------------------------------------
    X_all = build_feature_table(
        adata=adata,
        score_columns=score_columns,
        n_pcs=30,
    )
 
    y_all = adata.obs[LABEL_KEY].astype(str)
 
    X_all.head(50).to_csv(TABLE_DIR / "baseline_ml_feature_table_preview.csv")
 
    pd.DataFrame({"feature": X_all.columns.tolist()}).to_csv(
        TABLE_DIR / "baseline_ml_feature_columns.csv",
        index=False,
    )
 
    print(f"\nFeature matrix shape: {X_all.shape}")
 
    # ------------------------------------------------------------------
    # 3. Exclude low-confidence labels from training
    # ------------------------------------------------------------------
    train_mask = y_all != EXCLUDE_LABEL
 
    X = X_all.loc[train_mask].copy()
    y = y_all.loc[train_mask].copy()
 
    print("\nTraining label counts after excluding low-confidence spots:")
    print(y.value_counts())
 
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_names = label_encoder.classes_.tolist()
 
    label_mapping = pd.DataFrame(
        {
            "encoded_label": range(len(label_names)),
            "label_name": label_names,
        }
    )
 
    label_mapping.to_csv(
        TABLE_DIR / "baseline_ml_label_mapping.csv",
        index=False,
    )
 
    # ------------------------------------------------------------------
    # 4. Random stratified split
    # ------------------------------------------------------------------
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
    # 5. Train and compare five models
    # ------------------------------------------------------------------
    models = define_models()
 
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
 
    metrics_df = pd.DataFrame(all_metrics).sort_values(
        "macro_f1",
        ascending=False,
    )
 
    metrics_df.to_csv(
        TABLE_DIR / "baseline_ml_model_metrics.csv",
        index=False,
    )
 
    print("\nModel comparison:")
    print(metrics_df)
 
    plot_model_comparison(metrics_df)
 
    for model_name, fitted_model in fitted_models.items():
        plot_tree_feature_importance(
            model_name=model_name,
            model=fitted_model,
            feature_names=X.columns.tolist(),
            top_n=30,
        )
 
    # ------------------------------------------------------------------
    # 6. Select best model by macro F1
    # ------------------------------------------------------------------
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
 
    print(f"\nBest model by macro F1: {best_model_name}")
 
    model_path = MODEL_DIR / "baseline_spatial_niche_classifier.joblib"
    encoder_path = MODEL_DIR / "baseline_label_encoder.joblib"
    metadata_path = MODEL_DIR / "baseline_model_metadata.json"
 
    joblib.dump(best_model, model_path)
    joblib.dump(label_encoder, encoder_path)
 
    metadata = {
        "best_model": best_model_name,
        "label_key": LABEL_KEY,
        "excluded_label": EXCLUDE_LABEL,
        "n_all_spots": int(X_all.shape[0]),
        "n_training_spots": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": X.columns.tolist(),
        "label_names": label_names,
        "metrics": metrics_df.to_dict(orient="records"),
        "feature_groups": {
            "pca": "PC1-PC30",
            "qc": "available QC metrics",
            "spatial_coordinates": ["spatial_x", "spatial_y"],
            "marker_scores": score_columns,
            "squidpy_neighbor_marker_scores": [
                col for col in X.columns if col.startswith("neighbor_mean_")
            ],
        },
    }
 
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
 
    print(f"Saved best model to: {model_path}")
    print(f"Saved label encoder to: {encoder_path}")
    print(f"Saved model metadata to: {metadata_path}")
 
    # ------------------------------------------------------------------
    # 7. Predict every spot, including excluded/review spots
    # ------------------------------------------------------------------
    pred_encoded_all = best_model.predict(X_all)
 
    if hasattr(best_model, "predict_proba"):
        pred_proba_all = best_model.predict_proba(X_all)
        pred_confidence_all = pred_proba_all.max(axis=1)
    else:
        pred_confidence_all = np.full(X_all.shape[0], np.nan)
 
    pred_labels_all = label_encoder.inverse_transform(pred_encoded_all)
 
    adata.obs["baseline_ml_best_model"] = best_model_name
    adata.obs["baseline_ml_predicted_label"] = pd.Categorical(pred_labels_all)
    adata.obs["baseline_ml_prediction_confidence"] = pred_confidence_all
 
    prediction_table = adata.obs[
        [
            LABEL_KEY,
            "manual_niche_label_short",
            "manual_label_confidence",
            "baseline_ml_best_model",
            "baseline_ml_predicted_label",
            "baseline_ml_prediction_confidence",
        ]
    ].copy()
 
    prediction_table.to_csv(
        TABLE_DIR / "baseline_ml_spot_predictions.csv"
    )
 
    # ------------------------------------------------------------------
    # 8. Plot spatial predictions
    # ------------------------------------------------------------------
    plot_spatial_scatter(
        adata=adata,
        color=["baseline_ml_predicted_label"],
        filename="spatial_baseline_ml_predicted_labels.png",
        img=True,
        legend_loc="right margin",
    )
 
    plot_spatial_scatter(
        adata=adata,
        color=["baseline_ml_prediction_confidence"],
        filename="spatial_baseline_ml_prediction_confidence.png",
        img=True,
        legend_loc=None,
    )
 
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
    # 9. Save prediction-annotated object
    # ------------------------------------------------------------------
    adata.write_h5ad(OUTPUT_H5AD)
 
    print(f"\nSaved prediction-annotated AnnData object to: {OUTPUT_H5AD}")
    print("\nMilestone 5 complete.")
 
 
if __name__ == "__main__":
    main()
