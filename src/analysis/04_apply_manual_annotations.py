from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_marker_annotated.h5ad"
)

MANUAL_ANNOTATION_CSV = (
    PROJECT_ROOT
    / "data_manifest"
    / "annotations"
    / "leiden_r06_manual_cluster_annotations.csv"
)

OUTPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_final_labeled.h5ad"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "04_manual_annotations"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LIBRARY_ID = "Visium_Human_Breast_Cancer"
CLUSTER_KEY = "leiden_r06"


# Short names make legends easier to read.
SHORT_LABEL_BY_CLUSTER = {
    "0": "Myeloid/APC",
    "1": "B/plasma immune",
    "2": "Tumor epithelial",
    "3": "Tumor luminal-like",
    "4": "Hypoxic/metabolic tumor",
    "5": "Mixed epi/stromal",
    "6": "Mixed epi/stress",
    "7": "Mitochondrial/high oxidative",
    "8": "Keratin-high tumor",
    "9": "Luminal/secretory",
    "10": "Adipocyte/fat",
    "11": "Rare epithelial/VTCN1",
}


LOW_CONFIDENCE_TERMS = [
    "needs review",
    "low-confidence",
    "uncertain",
    "mixed",
    "mitochondrial",
]


def save_current_fig(filename: str) -> None:
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def classify_confidence(label: str, notes: str) -> str:
    text = f"{label} {notes}".lower()

    if any(term in text for term in LOW_CONFIDENCE_TERMS):
        return "low_confidence_or_review"

    return "high_confidence"


def main():
    print(f"Loading AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded AnnData object:")
    print(adata)

    print(f"\nReading manual annotations from: {MANUAL_ANNOTATION_CSV}")
    annotations = pd.read_csv(MANUAL_ANNOTATION_CSV)

    required_columns = [
        "cluster",
        "n_spots",
        "suggested_label",
        "top_10_de_markers",
        "manual_label",
        "notes",
    ]

    missing_columns = [
        col for col in required_columns if col not in annotations.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    annotations["cluster"] = annotations["cluster"].astype(str)
    annotations["manual_label"] = annotations["manual_label"].fillna("").astype(str)
    annotations["notes"] = annotations["notes"].fillna("").astype(str)

    empty_labels = annotations[annotations["manual_label"].str.strip() == ""]

    if not empty_labels.empty:
        raise ValueError(
            "Some clusters have empty manual_label values:\n"
            f"{empty_labels[['cluster', 'manual_label']]}"
        )

    clusters_in_adata = set(adata.obs[CLUSTER_KEY].astype(str).unique())
    clusters_in_annotations = set(annotations["cluster"].unique())

    missing_from_annotations = clusters_in_adata - clusters_in_annotations
    extra_in_annotations = clusters_in_annotations - clusters_in_adata

    if missing_from_annotations:
        raise ValueError(
            f"Clusters present in AnnData but missing from annotations: "
            f"{sorted(missing_from_annotations)}"
        )

    if extra_in_annotations:
        raise ValueError(
            f"Clusters present in annotations but missing from AnnData: "
            f"{sorted(extra_in_annotations)}"
        )

    annotations["manual_label_short"] = annotations["cluster"].map(
        SHORT_LABEL_BY_CLUSTER
    )

    missing_short_labels = annotations[
        annotations["manual_label_short"].isna()
    ]["cluster"].tolist()

    if missing_short_labels:
        raise ValueError(
            f"Missing short labels for clusters: {missing_short_labels}"
        )

    annotations["manual_label_confidence"] = annotations.apply(
        lambda row: classify_confidence(
            row["manual_label"],
            row["notes"],
        ),
        axis=1,
    )

    # For the first ML model, it is useful to have a training label that
    # excludes low-confidence or review clusters.
    annotations["ml_training_label"] = annotations.apply(
        lambda row: (
            row["manual_label_short"]
            if row["manual_label_confidence"] == "high_confidence"
            else "Exclude_low_confidence"
        ),
        axis=1,
    )

    print("\nManual annotation table:")
    print(
        annotations[
            [
                "cluster",
                "n_spots",
                "manual_label",
                "manual_label_short",
                "manual_label_confidence",
                "ml_training_label",
            ]
        ]
    )

    # Save a cleaned version of the annotations.
    cleaned_annotation_path = (
        TABLE_DIR / "leiden_r06_manual_cluster_annotations_cleaned.csv"
    )
    annotations.to_csv(cleaned_annotation_path, index=False)
    print(f"\nSaved cleaned annotation table to: {cleaned_annotation_path}")

    cluster_to_label = dict(
        zip(annotations["cluster"], annotations["manual_label"])
    )
    cluster_to_short_label = dict(
        zip(annotations["cluster"], annotations["manual_label_short"])
    )
    cluster_to_confidence = dict(
        zip(annotations["cluster"], annotations["manual_label_confidence"])
    )
    cluster_to_ml_label = dict(
        zip(annotations["cluster"], annotations["ml_training_label"])
    )
    cluster_to_notes = dict(
        zip(annotations["cluster"], annotations["notes"])
    )

    cluster_ids = adata.obs[CLUSTER_KEY].astype(str)

    adata.obs["manual_niche_label"] = (
        cluster_ids.map(cluster_to_label).astype("category")
    )

    adata.obs["manual_niche_label_short"] = (
        cluster_ids.map(cluster_to_short_label).astype("category")
    )

    adata.obs["manual_label_confidence"] = (
        cluster_ids.map(cluster_to_confidence).astype("category")
    )

    adata.obs["ml_training_label"] = (
        cluster_ids.map(cluster_to_ml_label).astype("category")
    )

    adata.obs["manual_annotation_notes"] = (
        cluster_ids.map(cluster_to_notes).astype(str)
    )

    adata.obs["cluster_manual_niche_label"] = (
        cluster_ids + ": " + adata.obs["manual_niche_label_short"].astype(str)
    ).astype("category")

    # Save spot-level label table.
    spot_label_table = adata.obs[
        [
            CLUSTER_KEY,
            "manual_niche_label",
            "manual_niche_label_short",
            "cluster_manual_niche_label",
            "manual_label_confidence",
            "ml_training_label",
            "manual_annotation_notes",
        ]
    ].copy()

    spot_label_path = TABLE_DIR / "spot_manual_niche_labels.csv"
    spot_label_table.to_csv(spot_label_path)
    print(f"Saved spot-level labels to: {spot_label_path}")

    # Cluster summary table.
    cluster_summary = (
        adata.obs[
            [
                CLUSTER_KEY,
                "manual_niche_label",
                "manual_niche_label_short",
                "manual_label_confidence",
                "ml_training_label",
            ]
        ]
        .groupby(
            [
                CLUSTER_KEY,
                "manual_niche_label",
                "manual_niche_label_short",
                "manual_label_confidence",
                "ml_training_label",
            ],
            observed=True,
        )
        .size()
        .reset_index(name="n_spots")
        .sort_values(CLUSTER_KEY)
    )

    cluster_summary_path = TABLE_DIR / "manual_niche_cluster_summary.csv"
    cluster_summary.to_csv(cluster_summary_path, index=False)
    print(f"Saved cluster summary to: {cluster_summary_path}")

    print("\nCluster summary:")
    print(cluster_summary)

    # Label counts for ML readiness.
    ml_counts = (
        adata.obs["ml_training_label"]
        .value_counts()
        .rename_axis("ml_training_label")
        .reset_index(name="n_spots")
    )

    ml_counts_path = TABLE_DIR / "ml_training_label_counts.csv"
    ml_counts.to_csv(ml_counts_path, index=False)
    print(f"Saved ML training label counts to: {ml_counts_path}")

    print("\nML training label counts:")
    print(ml_counts)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    sc.pl.spatial(
        adata,
        color=["cluster_manual_niche_label"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_cluster_manual_niche_labels.png")

    sc.pl.spatial(
        adata,
        color=["manual_niche_label_short"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_manual_niche_labels_short.png")

    sc.pl.umap(
        adata,
        color=["cluster_manual_niche_label"],
        legend_loc="right margin",
        frameon=False,
        show=False,
    )
    save_current_fig("umap_cluster_manual_niche_labels.png")

    sc.pl.umap(
        adata,
        color=["manual_label_confidence"],
        frameon=False,
        show=False,
    )
    save_current_fig("umap_manual_label_confidence.png")

    sc.pl.spatial(
        adata,
        color=["manual_label_confidence"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_manual_label_confidence.png")

    # Bar plot for final cluster labels.
    label_counts = (
        adata.obs["manual_niche_label_short"]
        .value_counts()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    label_counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Number of spots")
    ax.set_ylabel("Manual niche label")
    ax.set_title("Manual niche label counts")
    save_current_fig("manual_niche_label_counts.png")

    # Bar plot for ML labels.
    ml_label_counts = (
        adata.obs["ml_training_label"]
        .value_counts()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ml_label_counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Number of spots")
    ax.set_ylabel("ML training label")
    ax.set_title("ML training label counts")
    save_current_fig("ml_training_label_counts.png")

    # Save final labeled object.
    adata.write_h5ad(OUTPUT_H5AD)
    print(f"\nSaved final labeled AnnData object to: {OUTPUT_H5AD}")
    print("\nMilestone 3.5 complete.")


if __name__ == "__main__":
    main()
