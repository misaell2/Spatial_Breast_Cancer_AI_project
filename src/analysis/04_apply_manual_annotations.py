"""
Milestone 4: Apply manually curated biological niche annotations.

Purpose
-------
This script applies manually curated cluster annotations back to every Visium
spot in the AnnData object.

The previous marker-gene analysis script creates a cluster-level annotation
template. After manual biological review, this script converts:

    Leiden cluster -> curated biological label

into:

    every Visium spot -> curated biological niche label

Why this script exists
----------------------
This step is the bridge between biological interpretation and supervised ML.

Before this script:
    - clusters have marker genes
    - clusters have suggested labels
    - manual annotations exist in a CSV file
    - individual spots do not yet have final curated ML labels

After this script:
    - every spot has a final manual niche label
    - every spot has a short display label
    - every spot has a confidence category
    - low-confidence/review clusters are excluded from supervised ML training
    - the final labeled AnnData object can be used by modeling scripts

Why Squidpy is used here
------------------------
Squidpy is used for spatial visualization and optional spatial-neighborhood
analysis of the final manual niche labels.

This allows the project to ask:

    Which manually annotated niches are spatially adjacent or enriched near
    each other?

That is more biologically informative than plotting labels alone.

Input
-----
AnnData object from marker-gene analysis:

    data/processed/visium_human_breast_cancer_marker_annotated.h5ad

Manual annotation CSV:

    data_manifest/annotations/leiden_r06_manual_cluster_annotations.csv

Output
------
Final labeled AnnData object:

    data/processed/visium_human_breast_cancer_final_labeled.h5ad

Tables:

    results/tables/leiden_r06_manual_cluster_annotations_cleaned.csv
    results/tables/spot_manual_niche_labels.csv
    results/tables/manual_niche_cluster_summary.csv
    results/tables/ml_training_label_counts.csv
    results/tables/manual_niche_spatial_neighborhood_enrichment_zscore.csv
    results/tables/manual_niche_spatial_neighborhood_enrichment_count.csv

Figures:

    results/figures/04_manual_annotations/

Analysis notes
--------------
This script intentionally separates biological labels from ML training labels.

For example:

    manual_niche_label = "Rare neural/Schwann-like or edge-associated, low-confidence"

but:

    ml_training_label = "Exclude_low_confidence"

This prevents uncertain or review-needed labels from entering supervised ML
training.

Alternative choices
-------------------
Instead of excluding low-confidence clusters, future workflows could:

    - keep them as a separate "Uncertain/review" class
    - merge them into broader parent categories
    - use soft labels or probabilistic labels
    - use sample weights during training
    - train on high-confidence labels, then predict uncertain regions
    - add a manual confidence column instead of keyword-based confidence logic
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import squidpy as sq


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------
LIBRARY_ID = "Visium_Human_Breast_Cancer"
CLUSTER_KEY = "leiden_r06"

SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"


# ---------------------------------------------------------------------
# Short labels for plotting and ML readability
# ---------------------------------------------------------------------
# These short labels correspond to the new 10-cluster Squidpy-enabled run.
#
# Why this dictionary is explicit:
#   - it prevents inconsistent spelling across scripts
#   - it makes legends and ML reports easier to read
#   - it forces every cluster to be intentionally reviewed
SHORT_LABEL_BY_CLUSTER = {
    "0": "B/plasma immune",
    "1": "Hypoxic tumor",
    "2": "Myeloid/APC",
    "3": "Tumor epithelial",
    "4": "Luminal stress-like",
    "5": "Mixed epi/stromal",
    "6": "Luminal secretory",
    "7": "Keratin-high tumor",
    "8": "Rare epithelial/VTCN1",
    "9": "Rare neural-like",
}


# ---------------------------------------------------------------------
# Terms used to identify low-confidence or review-needed labels
# ---------------------------------------------------------------------
# These terms are searched in the manual_label and notes columns.
#
# Current behavior:
#   If any term is present, the cluster is marked as
#   "low_confidence_or_review" and excluded from supervised ML training.
#
# With the current 10-cluster annotations, this should exclude:
#   - cluster 5: Mixed epithelial/stromal-like, needs review
#   - cluster 8: Rare epithelial/immune-checkpoint-associated, needs review
#   - cluster 9: Rare neural/Schwann-like or edge-associated, low-confidence
LOW_CONFIDENCE_TERMS = [
    "needs review",
    "low-confidence",
    "uncertain",
    "mixed",
    "interpret cautiously",
    "exclude from ml",
]


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.

    Why this helper exists
    ----------------------
    This script generates several plots using Squidpy, Scanpy, and Matplotlib.
    Centralizing figure saving keeps output resolution and formatting
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
    ncols: int = 3,
    use_raw: bool | None = None,
) -> None:
    """
    Plot spatial data using Squidpy and save the figure.

    Why this helper exists
    ----------------------
    Squidpy is now the spatial plotting tool for this project. This wrapper
    keeps spatial plotting calls consistent and avoids passing unsupported
    arguments such as `show=False`.

    Parameters
    ----------
    color:
        AnnData `.obs` column, gene name, or list of columns/genes.

    filename:
        Output filename.

    img:
        Whether to show the tissue image under the spots.

    legend_loc:
        Legend location for categorical labels.

    ncols:
        Number of columns for multi-panel plots.

    use_raw:
        Whether expression should be read from `adata.raw` for gene plots.
    """
    kwargs = {
        "adata": adata,
        "color": color,
        "library_id": LIBRARY_ID,
        "img": img,
        "legend_loc": legend_loc,
        "ncols": ncols,
    }

    if use_raw is not None:
        kwargs["use_raw"] = use_raw

    sq.pl.spatial_scatter(**kwargs)
    save_current_fig(filename)


def classify_confidence(label: str, notes: str) -> str:
    """
    Classify whether a manually annotated cluster is high-confidence.

    Why this function exists
    ------------------------
    The ML model should not be trained on labels that the analyst already
    flagged as mixed, uncertain, low-confidence, or needing review.

    This function converts free-text annotation labels/notes into a reproducible
    confidence category.

    Alternative approaches
    ----------------------
    Future versions could use:
        - an explicit confidence column in the annotation CSV
        - numeric confidence scores from 0.0 to 1.0
        - high/medium/low confidence levels
        - manual include_in_ml flags
        - class-specific confidence thresholds
    """
    text = f"{label} {notes}".lower()

    if any(term in text for term in LOW_CONFIDENCE_TERMS):
        return "low_confidence_or_review"

    return "high_confidence"


def validate_inputs(adata, annotations: pd.DataFrame) -> None:
    """
    Validate that the AnnData object and annotation table match.

    Why this validation matters
    ---------------------------
    Manual labels are cluster-level labels. If the AnnData object has a
    different set of clusters than the CSV file, labels could be incorrectly
    assigned or missing.

    This can happen after rerunning preprocessing or changing Leiden resolution.
    """
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
        raise ValueError(f"Missing required annotation columns: {missing_columns}")

    if CLUSTER_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find {CLUSTER_KEY} in adata.obs")

    clusters_in_adata = set(adata.obs[CLUSTER_KEY].astype(str).unique())
    clusters_in_annotations = set(annotations["cluster"].astype(str).unique())

    missing_from_annotations = clusters_in_adata - clusters_in_annotations
    extra_in_annotations = clusters_in_annotations - clusters_in_adata

    if missing_from_annotations:
        raise ValueError(
            "Clusters present in AnnData but missing from annotations: "
            f"{sorted(missing_from_annotations)}"
        )

    if extra_in_annotations:
        raise ValueError(
            "Clusters present in annotations but missing from AnnData: "
            f"{sorted(extra_in_annotations)}"
        )

    missing_short_labels = sorted(clusters_in_adata - set(SHORT_LABEL_BY_CLUSTER))

    if missing_short_labels:
        raise ValueError(
            f"Missing short labels for clusters: {missing_short_labels}"
        )


def prepare_annotation_table(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the manual annotation table and add derived label columns.

    Added columns
    -------------
    manual_label_short:
        Short label used for plots and ML reports.

    manual_label_confidence:
        high_confidence or low_confidence_or_review.

    ml_training_label:
        Final supervised ML label. Low-confidence/review clusters are assigned
        Exclude_low_confidence.

    Why this step exists
    --------------------
    The original manual CSV captures biological reasoning. This cleaned version
    adds the structured columns needed for plotting and ML.
    """
    annotations = annotations.copy()

    annotations["cluster"] = annotations["cluster"].astype(str)
    annotations["manual_label"] = annotations["manual_label"].fillna("").astype(str)
    annotations["notes"] = annotations["notes"].fillna("").astype(str)

    empty_labels = annotations[annotations["manual_label"].str.strip() == ""]

    if not empty_labels.empty:
        raise ValueError(
            "Some clusters have empty manual_label values:\n"
            f"{empty_labels[['cluster', 'manual_label']]}"
        )

    annotations["manual_label_short"] = annotations["cluster"].map(
        SHORT_LABEL_BY_CLUSTER
    )

    annotations["manual_label_confidence"] = annotations.apply(
        lambda row: classify_confidence(row["manual_label"], row["notes"]),
        axis=1,
    )

    annotations["ml_training_label"] = annotations.apply(
        lambda row: (
            row["manual_label_short"]
            if row["manual_label_confidence"] == "high_confidence"
            else "Exclude_low_confidence"
        ),
        axis=1,
    )

    return annotations


def apply_annotations_to_spots(adata, annotations: pd.DataFrame):
    """
    Map cluster-level annotations to every spot in AnnData.

    Why this function exists
    ------------------------
    Manual labels are curated at the cluster level, but ML training and spatial
    plotting require spot-level labels.

    This function propagates:

        cluster -> manual niche label

    to every spot in that cluster.
    """
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

    return adata


def save_summary_tables(adata, annotations: pd.DataFrame) -> None:
    """
    Save cluster-level, spot-level, and ML-label summary tables.

    Why these tables matter
    -----------------------
    They make the annotation process transparent and allow downstream scripts to
    verify which labels were used for ML training.
    """
    cleaned_annotation_path = (
        TABLE_DIR / "leiden_r06_manual_cluster_annotations_cleaned.csv"
    )
    annotations.to_csv(cleaned_annotation_path, index=False)
    print(f"Saved cleaned annotation table to: {cleaned_annotation_path}")

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


def run_manual_label_neighborhood_enrichment(adata) -> None:
    """
    Run Squidpy neighborhood enrichment on manual niche labels.

    Why this analysis is useful
    ---------------------------
    This asks whether manually annotated biological niches are spatially
    adjacent more often than expected by chance.

    This can support biological interpretation, for example:

        - hypoxic tumor regions near tumor epithelial regions
        - immune niches enriched near tumor-associated tissue
        - adipose/edge-like regions separated from core tumor regions

    Important note
    --------------
    Low-confidence labels are still included in this spatial analysis because
    spatial context can help interpret uncertain regions. They are excluded only
    from supervised ML training.
    """
    if SPATIAL_CONNECTIVITY_KEY not in adata.obsp:
        print(
            f"\nSkipping manual label neighborhood enrichment: "
            f"{SPATIAL_CONNECTIVITY_KEY} not found in adata.obsp."
        )
        return

    print("\nRunning Squidpy neighborhood enrichment for manual niche labels...")

    label_key = "manual_niche_label_short"
    adata.obs[label_key] = adata.obs[label_key].astype("category")

    sq.gr.nhood_enrichment(
        adata,
        cluster_key=label_key,
        n_perms=1000,
        seed=42,
        n_jobs=1,
        show_progress_bar=True,
    )

    result_key = f"{label_key}_nhood_enrichment"

    if result_key not in adata.uns:
        print(f"Warning: expected {result_key} in adata.uns but did not find it.")
        return

    enrichment = adata.uns[result_key]
    categories = adata.obs[label_key].cat.categories.astype(str).tolist()

    if "zscore" in enrichment:
        zscore_df = pd.DataFrame(
            enrichment["zscore"],
            index=categories,
            columns=categories,
        )
        zscore_path = (
            TABLE_DIR / "manual_niche_spatial_neighborhood_enrichment_zscore.csv"
        )
        zscore_df.to_csv(zscore_path)
        print(f"Saved manual niche neighborhood z-scores to: {zscore_path}")

    if "count" in enrichment:
        count_df = pd.DataFrame(
            enrichment["count"],
            index=categories,
            columns=categories,
        )
        count_path = (
            TABLE_DIR / "manual_niche_spatial_neighborhood_enrichment_count.csv"
        )
        count_df.to_csv(count_path)
        print(f"Saved manual niche neighborhood counts to: {count_path}")

    sq.pl.nhood_enrichment(
        adata,
        cluster_key=label_key,
        annotate=True,
        method="average",
        figsize=(9, 9),
    )
    save_current_fig("manual_niche_spatial_neighborhood_enrichment.png")


def plot_annotation_outputs(adata) -> None:
    """
    Generate spatial, UMAP, and label-count plots.

    Why these plots are included
    ----------------------------
    These figures confirm whether manual labels are spatially coherent, whether
    low-confidence labels are localized, and how much data will be used for ML.
    """
    plot_spatial_scatter(
        adata,
        color=["cluster_manual_niche_label"],
        filename="spatial_cluster_manual_niche_labels.png",
        img=True,
        legend_loc="right margin",
    )

    plot_spatial_scatter(
        adata,
        color=["manual_niche_label_short"],
        filename="spatial_manual_niche_labels_short.png",
        img=True,
        legend_loc="right margin",
    )

    plot_spatial_scatter(
        adata,
        color=["manual_label_confidence"],
        filename="spatial_manual_label_confidence.png",
        img=True,
        legend_loc="right margin",
    )

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


def main() -> None:
    """
    Apply manual annotations to the marker-annotated AnnData object.

    Workflow
    --------
    1. Load marker-annotated AnnData object.
    2. Load manual annotation CSV.
    3. Validate that annotation clusters match AnnData clusters.
    4. Add short labels, confidence labels, and ML training labels.
    5. Map cluster-level annotations to every spot.
    6. Save summary tables.
    7. Run Squidpy neighborhood enrichment on manual labels.
    8. Save spatial, UMAP, and count plots.
    9. Save final labeled AnnData object.
    """
    print(f"Loading AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded AnnData object:")
    print(adata)

    print(f"\nReading manual annotations from: {MANUAL_ANNOTATION_CSV}")
    annotations = pd.read_csv(MANUAL_ANNOTATION_CSV)

    validate_inputs(adata, annotations)

    annotations = prepare_annotation_table(annotations)

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

    adata = apply_annotations_to_spots(adata, annotations)

    save_summary_tables(adata, annotations)

    run_manual_label_neighborhood_enrichment(adata)

    plot_annotation_outputs(adata)

    adata.write_h5ad(OUTPUT_H5AD)

    print(f"\nSaved final labeled AnnData object to: {OUTPUT_H5AD}")
    print("\nMilestone 4 complete.")


if __name__ == "__main__":
    main()
