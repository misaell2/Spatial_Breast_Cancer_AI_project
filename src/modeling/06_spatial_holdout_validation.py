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

LIBRARY_ID = "Visium_Human_Breast_Cancer"
LABEL_KEY = "ml_training_label"
EXCLUDE_LABEL = "Exclude_low_confidence"
SPATIAL_BLOCK_KEY = "spatial_block_3x3"


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
    """Make a string safe for filenames."""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_\\-]+", "_", text)
    return text.strip("_")


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


def add_spatial_blocks(adata, n_bins_x=3, n_bins_y=3):
    """Divide tissue coordinates into spatial grid blocks."""
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


def build_feature_table(adata, score_columns, n_pcs=30):
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

    features_with_spatial = pd.concat(
        [
            pc_df,
            qc_df,
            spatial_df,
            signature_df,
        ],
        axis=1,
    )

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
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median(numeric_only=True))
        cleaned_feature_sets[name] = features

    return cleaned_feature_sets


def plot_confusion_matrix(cm, labels, title, filename):
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


def plot_spatial_blocks(adata):
    sc.pl.spatial(
        adata,
        color=[SPATIAL_BLOCK_KEY, LABEL_KEY],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_blocks_and_training_labels.png")


def summarize_block_labels(adata):
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


def train_and_evaluate_holdout(X, y, blocks, heldout_block, feature_set_name):
    test_mask = blocks == heldout_block
    train_mask = ~test_mask

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    if X_test.shape[0] < 30:
        print(f"  Skipping {heldout_block}: fewer than 30 test spots.")
        return None

    if y_test.nunique() < 2:
        print(f"  Skipping {heldout_block}: fewer than 2 labels in test block.")
        return None

    if y_train.nunique() < 2:
        print(f"  Skipping {heldout_block}: fewer than 2 labels in training set.")
        return None

    train_labels = set(y_train.unique())
    test_labels = set(y_test.unique())

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
        filename=f"spatial_holdout_{safe_name(feature_set_name)}_{safe_name(heldout_block)}_confusion_matrix.png",
    )

    return metrics


def run_spatial_holdout_validation(feature_sets, y, blocks):
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


def plot_holdout_metrics(metrics_df):
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

    summary = (
        metrics_df.groupby("feature_set")[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]]
        .agg(["mean", "std", "min", "max"])
    )

    summary_path = TABLE_DIR / "spatial_holdout_validation_summary_by_feature_set.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary metrics to: {summary_path}")

    print("\nSpatial holdout summary by feature set:")
    print(summary)


def main():
    print(f"Loading final labeled AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded object:")
    print(adata)

    if LABEL_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find label column: {LABEL_KEY}")

    print("\nOriginal ML label counts:")
    print(adata.obs[LABEL_KEY].astype(str).value_counts())

    adata, score_columns = add_marker_signature_scores(adata)
    adata = add_spatial_blocks(adata, n_bins_x=3, n_bins_y=3)

    print("\nSpatial block counts:")
    print(adata.obs[SPATIAL_BLOCK_KEY].astype(str).value_counts().sort_index())

    summarize_block_labels(adata)
    plot_spatial_blocks(adata)

    feature_sets_all = build_feature_table(
        adata,
        score_columns=score_columns,
        n_pcs=30,
    )

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
