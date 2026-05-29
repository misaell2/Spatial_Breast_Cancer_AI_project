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

LIBRARY_ID = "Visium_Human_Breast_Cancer"
LABEL_KEY = "ml_training_label"
EXCLUDE_LABEL = "Exclude_low_confidence"


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
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def get_available_genes(adata, genes):
    if adata.raw is not None:
        available = set(adata.raw.var_names)
    else:
        available = set(adata.var_names)

    return [gene for gene in genes if gene in available]


def add_marker_signature_scores(adata):
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


def build_feature_table(adata, score_columns, n_pcs=30):
    if "X_pca" not in adata.obsm:
        raise ValueError("Expected PCA coordinates in adata.obsm['X_pca'].")

    pcs = adata.obsm["X_pca"][:, :n_pcs]
    pc_cols = [f"PC{i + 1}" for i in range(pcs.shape[1])]
    pc_df = pd.DataFrame(pcs, index=adata.obs_names, columns=pc_cols)

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

    features = pd.concat(
        [
            pc_df,
            qc_df,
            spatial_df,
            signature_df,
        ],
        axis=1,
    )

    # Clean any unexpected infinities or missing values.
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))

    return features


def plot_confusion_matrix(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(9, 8))
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


def plot_feature_importance(model, feature_names, filename, top_n=30):
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

    importance_df = (
        importances
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importances to: {importance_path}")


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, label_names):
    print(f"\nTraining model: {model_name}")
    model.fit(X_train, y_train)

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

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm,
        labels=label_names,
        title=f"{model_name} confusion matrix",
        filename=f"{model_name}_confusion_matrix.png",
    )

    return metrics, model


def main():
    print(f"Loading final labeled AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded object:")
    print(adata)

    if LABEL_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find label column: {LABEL_KEY}")

    print("\nOriginal ML label counts:")
    print(adata.obs[LABEL_KEY].value_counts())

    # ------------------------------------------------------------------
    # 1. Add marker signature scores
    # ------------------------------------------------------------------
    adata, score_columns = add_marker_signature_scores(adata)

    print("\nMarker signature score columns:")
    print(score_columns)

    # ------------------------------------------------------------------
    # 2. Build feature matrix
    # ------------------------------------------------------------------
    X_all = build_feature_table(
        adata,
        score_columns=score_columns,
        n_pcs=30,
    )

    y_all = adata.obs[LABEL_KEY].astype(str)

    feature_table_path = TABLE_DIR / "baseline_ml_feature_table_preview.csv"
    X_all.head(50).to_csv(feature_table_path)
    print(f"\nSaved feature table preview to: {feature_table_path}")

    # ------------------------------------------------------------------
    # 3. Exclude low-confidence labels for supervised training
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

    label_mapping_path = TABLE_DIR / "baseline_ml_label_mapping.csv"
    label_mapping.to_csv(label_mapping_path, index=False)
    print(f"Saved label mapping to: {label_mapping_path}")

    # ------------------------------------------------------------------
    # 4. Stratified train/test split
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
    # 5. Define baseline models
    # ------------------------------------------------------------------
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    multi_class="auto",
                    solver="lbfgs",
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

    # Feature importance for random forest.
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
    model_path = MODEL_DIR / "baseline_spatial_niche_classifier.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")

    encoder_path = MODEL_DIR / "baseline_label_encoder.joblib"
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved label encoder to: {encoder_path}")

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
    sc.pl.spatial(
        adata,
        color=["baseline_ml_predicted_label"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_baseline_ml_predicted_labels.png")

    sc.pl.spatial(
        adata,
        color=["baseline_ml_prediction_confidence"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_baseline_ml_prediction_confidence.png")

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
    adata.write_h5ad(OUTPUT_H5AD)
    print(f"\nSaved prediction-annotated AnnData object to: {OUTPUT_H5AD}")

    print("\nMilestone 4 complete.")


if __name__ == "__main__":
    main()
