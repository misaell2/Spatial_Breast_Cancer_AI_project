"""
Milestone 3.5: Apply manually curated biological niche annotations.

Purpose
-------
This script takes the manually reviewed cluster annotation table and applies
those annotations back to every Visium spot in the AnnData object.

This step is important because the previous marker-gene analysis produces
cluster-level evidence, but the downstream machine learning model needs a
spot-level label column.

In other words, this script converts:

    Leiden cluster -> manually curated biological label

into:

    every spatial spot -> manually curated biological niche label

Why this script exists
----------------------
The manual annotation table is the bridge between biological interpretation
and supervised ML.

Before this script:
    - clusters have marker genes
    - clusters have manually reviewed labels in a CSV file
    - individual spots do not yet have final curated ML labels

After this script:
    - every spot has a final biological niche label
    - every spot has a short display label for plotting
    - every spot has a confidence label
    - low-confidence clusters are excluded from supervised ML training
    - the final labeled AnnData object can be used by modeling scripts

Input
-----
AnnData object from marker gene analysis:

    data/processed/visium_human_breast_cancer_marker_annotated.h5ad

Manual annotation CSV:

    data_manifest/annotations/leiden_r06_manual_cluster_annotations.csv

The manual annotation CSV is expected to contain at least:

    cluster
    n_spots
    suggested_label
    top_10_de_markers
    manual_label
    notes

Output
------
Final labeled AnnData object:

    data/processed/visium_human_breast_cancer_final_labeled.h5ad

Tables:

    results/tables/leiden_r06_manual_cluster_annotations_cleaned.csv
    results/tables/spot_manual_niche_labels.csv
    results/tables/manual_niche_cluster_summary.csv
    results/tables/ml_training_label_counts.csv

Figures:

    results/figures/04_manual_annotations/

Analysis notes
--------------
This script intentionally separates biological labels from ML training labels.

For example:
    manual_niche_label = "Mitochondrial/high oxidative signal, low-confidence"

but:

    ml_training_label = "Exclude_low_confidence"

This prevents the baseline ML model from learning labels that were explicitly
marked as uncertain, mixed, or low-confidence.

Alternative choices
-------------------
Instead of excluding low-confidence clusters, future workflows could:

    - keep them as a separate "Uncertain/mixed" class
    - merge them into broader parent categories
    - use soft labels or probabilistic labels
    - weight high-confidence spots more heavily during training
    - train a first model only on high-confidence labels, then predict labels
      for excluded regions
    - create a three-level confidence system instead of binary confidence
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This script lives in:
#   src/analysis/04_apply_manual_annotations.py
#
# parents[2] moves from:
#   src/analysis/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input object produced by:
#   src/analysis/03_marker_gene_analysis.py
#
# This file contains marker gene analysis results and suggested labels, but
# not yet the final manually curated niche labels.
INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_marker_annotated.h5ad"
)

# Manually curated annotation table.
#
# This file is intentionally stored in data_manifest/annotations instead of
# results/ because it is a human-curated project artifact, not just a generated
# output. It should be version-controlled.
MANUAL_ANNOTATION_CSV = (
    PROJECT_ROOT
    / "data_manifest"
    / "annotations"
    / "leiden_r06_manual_cluster_annotations.csv"
)

# Final labeled AnnData object.
#
# This becomes the main input to the baseline ML model.
OUTPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_final_labeled.h5ad"
)

# Output folders for figures and summary tables.
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "04_manual_annotations"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset and cluster metadata
# ---------------------------------------------------------------------
# This should match the key in adata.uns["spatial"].
# Scanpy uses this to find the matching Visium image and scale factors.
LIBRARY_ID = "Visium_Human_Breast_Cancer"

# Leiden cluster key created in preprocessing.
# All manual annotations are mapped to spots through this cluster column.
CLUSTER_KEY = "leiden_r06"


# ---------------------------------------------------------------------
# Short labels for plotting and ML readability
# ---------------------------------------------------------------------
# The manual labels can be biologically descriptive and long.
# Short labels make legends, bar plots, confusion matrices, and ML reports
# easier to read.
#
# Why this dictionary is explicit:
#   - It forces every cluster to have a clean display label.
#   - It avoids inconsistent spelling across scripts.
#   - It makes model classes more readable.
#
# Alternative:
#   These short labels could be stored directly in the manual annotation CSV
#   as a "manual_label_short" column. The current approach keeps the original
#   manual CSV focused on biological notes and lets this script define the
#   display/model-friendly labels.
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


# ---------------------------------------------------------------------
# Terms used to identify low-confidence or review-needed labels
# ---------------------------------------------------------------------
# These terms are searched in the manual label and notes columns.
# If any term is present, the cluster is marked as low-confidence/review.
#
# Why this approach is useful:
#   - It makes confidence assignment reproducible.
#   - It allows the manual notes to control whether a label should be used for
#     ML training.
#   - It prevents uncertain labels from accidentally entering the supervised
#     training set.
#
# Alternative parameters:
#   - Add "rare" if all rare clusters should be excluded.
#   - Add "technical" if suspected technical artifacts are annotated.
#   - Add "edge" if edge-effect clusters are annotated.
#   - Use a manual CSV column named "confidence" instead of keyword matching.
LOW_CONFIDENCE_TERMS = [
    "needs review",
    "low-confidence",
    "uncertain",
    "mixed",
    "mitochondrial",
]


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.

    Parameters
    ----------
    filename:
        Name of the figure file to save inside FIGURE_DIR.

    Why this helper exists
    ----------------------
    Many plotting calls are made in this script. Centralizing the save logic
    ensures all figures use the same resolution, bounding box behavior, and
    output directory.

    Alternative parameters:
        dpi=150  for smaller files
        dpi=300  for publication/GitHub-quality figures
        dpi=600  for print-quality figures, but larger files
    """
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def classify_confidence(label: str, notes: str) -> str:
    """
    Classify whether a manually annotated cluster is high-confidence.

    Parameters
    ----------
    label:
        Manual biological label for the cluster.

    notes:
        Manual notes explaining the annotation.

    Returns
    -------
    str
        "high_confidence" or "low_confidence_or_review"

    Why this function exists
    ------------------------
    The ML model should not be trained on labels that the analyst already
    flagged as mixed, uncertain, low-confidence, or potentially technical.

    This function converts free-text biological notes into a reproducible
    confidence category.

    Alternative approaches
    ----------------------
    Instead of keyword matching, future versions could use:
        - an explicit "confidence" column in the annotation CSV
        - confidence scores such as 0.0 to 1.0
        - categories such as high, medium, low
        - manual inclusion/exclusion flags per cluster
        - class-specific confidence thresholds
    """
    # Combine label and notes so that either field can flag low confidence.
    text = f"{label} {notes}".lower()

    # If any low-confidence keyword is present, exclude the cluster from the
    # high-confidence supervised training set.
    if any(term in text for term in LOW_CONFIDENCE_TERMS):
        return "low_confidence_or_review"

    return "high_confidence"


def main() -> None:
    """
    Apply manual cluster labels to every Visium spot.

    Workflow
    --------
    1. Load marker-annotated AnnData object.
    2. Read manually curated cluster annotation CSV.
    3. Validate that required annotation columns are present.
    4. Validate that every Leiden cluster has exactly one annotation.
    5. Create short labels and confidence labels.
    6. Create ML training labels that exclude low-confidence clusters.
    7. Map cluster-level annotations to spot-level labels.
    8. Save summary tables and plots.
    9. Save final labeled AnnData object.
    """
    print(f"Loading AnnData object from: {INPUT_H5AD}")

    # Load the object created by marker gene analysis.
    # This object already contains Leiden clusters and marker-gene outputs.
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded AnnData object:")
    print(adata)

    print(f"\nReading manual annotations from: {MANUAL_ANNOTATION_CSV}")

    # Read the manually curated CSV.
    # This file is expected to be version-controlled because it captures the
    # human biological interpretation of each cluster.
    annotations = pd.read_csv(MANUAL_ANNOTATION_CSV)

    # ------------------------------------------------------------------
    # 1. Validate required columns
    # ------------------------------------------------------------------
    # These columns are required because the script needs:
    #   - cluster IDs to map labels back to AnnData
    #   - manual labels to define biological niches
    #   - notes to classify low-confidence labels
    #   - top markers/suggested labels for traceability
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

    # Fail early if the CSV format is wrong.
    # This prevents silent mislabeling of spots.
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Standardize cluster IDs to strings.
    #
    # Why strings?
    #   AnnData categorical labels often behave more predictably as strings,
    #   and dictionary mapping is less error-prone when all cluster IDs share
    #   the same type.
    annotations["cluster"] = annotations["cluster"].astype(str)

    # Replace missing label/note values with empty strings so downstream string
    # operations are safe.
    annotations["manual_label"] = annotations["manual_label"].fillna("").astype(str)
    annotations["notes"] = annotations["notes"].fillna("").astype(str)

    # ------------------------------------------------------------------
    # 2. Validate that every cluster has a manual label
    # ------------------------------------------------------------------
    # Empty manual labels would lead to unlabeled spots or incorrect ML labels.
    empty_labels = annotations[annotations["manual_label"].str.strip() == ""]

    if not empty_labels.empty:
        raise ValueError(
            "Some clusters have empty manual_label values:\n"
            f"{empty_labels[['cluster', 'manual_label']]}"
        )

    # ------------------------------------------------------------------
    # 3. Validate agreement between AnnData clusters and annotation CSV
    # ------------------------------------------------------------------
    # The annotation CSV should contain exactly the clusters present in AnnData.
    # This protects against:
    #   - rerunning Leiden at a different resolution without updating labels
    #   - missing annotations
    #   - extra annotations from a previous run
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

    # ------------------------------------------------------------------
    # 4. Add short plotting/ML labels
    # ------------------------------------------------------------------
    # These are shorter than the full manual labels and therefore better for:
    #   - plot legends
    #   - confusion matrices
    #   - classification reports
    #   - README figures
    annotations["manual_label_short"] = annotations["cluster"].map(
        SHORT_LABEL_BY_CLUSTER
    )

    # Fail if any cluster lacks a short label.
    # This forces the analyst to consciously decide how each cluster should be
    # represented in plots and ML outputs.
    missing_short_labels = annotations[
        annotations["manual_label_short"].isna()
    ]["cluster"].tolist()

    if missing_short_labels:
        raise ValueError(
            f"Missing short labels for clusters: {missing_short_labels}"
        )

    # ------------------------------------------------------------------
    # 5. Add confidence labels
    # ------------------------------------------------------------------
    # Confidence is assigned from the manual label and notes text.
    # This is used to decide whether a cluster should be included in the
    # supervised ML training set.
    annotations["manual_label_confidence"] = annotations.apply(
        lambda row: classify_confidence(
            row["manual_label"],
            row["notes"],
        ),
        axis=1,
    )

    # ------------------------------------------------------------------
    # 6. Create ML training labels
    # ------------------------------------------------------------------
    # High-confidence clusters keep their short biological label.
    # Low-confidence/review clusters are grouped as Exclude_low_confidence.
    #
    # Why exclude low-confidence labels?
    #   The first ML model should learn from labels that are biologically
    #   defensible. Training on uncertain labels could make performance look
    #   better while reducing scientific validity.
    #
    # Alternative approaches:
    #   - Keep uncertain clusters as an "Uncertain" class.
    #   - Collapse mixed clusters into broader labels.
    #   - Use sample weights to downweight uncertain clusters.
    #   - Train a semi-supervised model using high-confidence labels only.
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

    # ------------------------------------------------------------------
    # 7. Save cleaned annotation table
    # ------------------------------------------------------------------
    # This table captures the final interpreted cluster labels after adding
    # short labels, confidence categories, and ML training labels.
    cleaned_annotation_path = (
        TABLE_DIR / "leiden_r06_manual_cluster_annotations_cleaned.csv"
    )
    annotations.to_csv(cleaned_annotation_path, index=False)
    print(f"\nSaved cleaned annotation table to: {cleaned_annotation_path}")

    # ------------------------------------------------------------------
    # 8. Build mapping dictionaries
    # ------------------------------------------------------------------
    # These dictionaries map:
    #   cluster ID -> annotation field
    #
    # They are used to propagate cluster-level annotations to every spot.
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

    # Convert the AnnData cluster column to string so it matches dictionary keys.
    cluster_ids = adata.obs[CLUSTER_KEY].astype(str)

    # ------------------------------------------------------------------
    # 9. Add spot-level annotation columns to AnnData
    # ------------------------------------------------------------------
    # Every spot gets the annotation assigned to its Leiden cluster.
    #
    # These columns serve different downstream purposes:
    #
    # manual_niche_label:
    #   Full biological label.
    #
    # manual_niche_label_short:
    #   Short label for plots and ML classes.
    #
    # manual_label_confidence:
    #   Whether the label is high-confidence or should be reviewed/excluded.
    #
    # ml_training_label:
    #   Final target label used by ML scripts.
    #
    # manual_annotation_notes:
    #   Biological reasoning from the annotation table.
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

    # Combined cluster + short label is useful for plots where preserving the
    # original Leiden cluster ID matters.
    adata.obs["cluster_manual_niche_label"] = (
        cluster_ids + ": " + adata.obs["manual_niche_label_short"].astype(str)
    ).astype("category")

    # ------------------------------------------------------------------
    # 10. Save spot-level label table
    # ------------------------------------------------------------------
    # This table is useful for:
    #   - debugging label propagation
    #   - checking ML target labels
    #   - linking every spot barcode to its final biological annotation
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

    # ------------------------------------------------------------------
    # 11. Save cluster-level summary table
    # ------------------------------------------------------------------
    # This table summarizes how many spots belong to each manually annotated
    # niche. It is useful for README summaries and for diagnosing class
    # imbalance before ML modeling.
    #
    # Alternative:
    #   Add percentages by dividing n_spots by total spots.
    #   Add high-confidence-only summaries for ML-specific reporting.
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

    # ------------------------------------------------------------------
    # 12. Save ML training label counts
    # ------------------------------------------------------------------
    # This shows how many spots will be available for each model class.
    #
    # Why this matters:
    #   Class imbalance affects model performance, confusion matrices, and the
    #   interpretation of accuracy. Macro F1 is later used because it treats
    #   classes more equally than raw accuracy.
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
    # 13. Figures: spatial and UMAP label inspection
    # ------------------------------------------------------------------
    # These plots confirm whether manual labels:
    #   - form coherent tissue regions
    #   - align with UMAP/expression structure
    #   - identify which regions were excluded from ML training
    #
    # Alternative plotting choices:
    #   - use Squidpy spatial plotting if available
    #   - generate one figure per label for publication-style panels
    #   - save SVG/PDF versions for vector graphics
    #   - adjust spot size/alpha for dense plots
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

    # ------------------------------------------------------------------
    # 14. Figure: manual niche label counts
    # ------------------------------------------------------------------
    # Bar plots make class imbalance easy to inspect.
    #
    # Why horizontal bars?
    #   Labels can be long, and horizontal bars are more readable.
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

    # ------------------------------------------------------------------
    # 15. Figure: ML training label counts
    # ------------------------------------------------------------------
    # This plot explicitly shows which labels will be used for training and how
    # many spots are excluded as low-confidence.
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

    # ------------------------------------------------------------------
    # 16. Save final labeled AnnData object
    # ------------------------------------------------------------------
    # This object is the main input for:
    #   src/modeling/05_train_baseline_niche_classifier.py
    #   src/modeling/06_spatial_holdout_validation.py
    #
    # It preserves the expression data, embeddings, spatial information, and
    # final labels needed for ML.
    adata.write_h5ad(OUTPUT_H5AD)

    print(f"\nSaved final labeled AnnData object to: {OUTPUT_H5AD}")
    print("\nMilestone 3.5 complete.")


if __name__ == "__main__":
    main()
