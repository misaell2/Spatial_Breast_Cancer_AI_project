"""
Milestone 2: Preprocess, cluster, and add preliminary marker-score labels.

Purpose
-------
This script takes the raw-QC AnnData object from Milestone 1 and performs the
main preprocessing and unsupervised clustering workflow.

It also adds a first-pass marker-score interpretation layer so that Leiden
clusters are easier to interpret before formal marker-gene analysis.

Input
-----
    data/processed/visium_human_breast_cancer_raw_qc.h5ad

Outputs
-------
Processed AnnData object:

    data/processed/visium_human_breast_cancer_processed_clustered.h5ad

Figures:

    results/figures/02_preprocessing_clustering/

Tables:

    results/tables/

Main analysis steps
-------------------
1. Load raw-QC AnnData object.
2. Apply conservative QC filtering.
3. Preserve raw counts.
4. Normalize counts and log-transform expression.
5. Select highly variable genes.
6. Run scaling, PCA, nearest-neighbor graph construction, UMAP, and Leiden clustering.
7. Plot QC, clustering, and marker genes.
8. Score biological marker-gene signatures.
9. Assign preliminary biological labels to Leiden clusters.
10. Save processed AnnData object.

Analysis notes
--------------
The preliminary marker-score labels are intentionally conservative. They are
useful for quick interpretation, but the more rigorous biological annotation is
performed later in:

    src/analysis/03_marker_gene_analysis.py

Possible alternatives
---------------------
- QC thresholds could be adjusted after visual inspection.
- HVG selection could use `flavor="seurat_v3"` if `scikit-misc` is installed.
- Leiden resolution could be tuned, e.g. 0.3, 0.6, 0.8, 1.0.
- Clustering could also use Louvain or graph-based domain methods.
- Marker scores could be replaced or complemented by AUCell, ssGSEA, GSVA,
  decoupler, or single-cell-reference deconvolution.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_raw_qc.h5ad"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "02_preprocessing_clustering"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------
LIBRARY_ID = "Visium_Human_Breast_Cancer"
CLUSTER_KEY = "leiden_r06"


# ---------------------------------------------------------------------
# Marker sets for preliminary cluster interpretation
# ---------------------------------------------------------------------
MARKER_SETS = {
    "tumor_epithelial": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "MUC1",
        "KRT7",
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
    ],
    "t_cell": [
        "CD3D",
        "CD3E",
        "CD2",
        "TRAC",
        "CD8A",
        "CD8B",
    ],
    "b_cell": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
    ],
    "myeloid_macrophage": [
        "LST1",
        "C1QA",
        "C1QB",
        "C1QC",
        "CD68",
        "TYROBP",
        "FCER1G",
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
        "STMN1",
    ],
}


LABEL_DISPLAY_NAMES = {
    "tumor_epithelial": "Tumor epithelial",
    "stromal_caf": "Stromal/CAF",
    "pan_immune": "Immune-enriched",
    "t_cell": "T-cell enriched",
    "b_cell": "B-cell enriched",
    "myeloid_macrophage": "Myeloid/macrophage",
    "endothelial": "Endothelial/vascular",
    "proliferation": "Proliferative",
}


def save_current_fig(filename: str) -> None:
    """Save the active matplotlib figure and close it."""
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def get_available_markers(adata, genes: list[str]) -> list[str]:
    """
    Return marker genes available in the AnnData object.

    This uses `adata.raw` when available because the object is later subset to
    highly variable genes for PCA/clustering, while `adata.raw` preserves the
    full normalized expression matrix.
    """
    if adata.raw is not None:
        gene_index = set(adata.raw.var_names)
    else:
        gene_index = set(adata.var_names)

    return [gene for gene in genes if gene in gene_index]


def assign_cluster_labels(cluster_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Assign preliminary biological labels to Leiden clusters.

    The label is based on the highest mean marker signature score per cluster,
    with simple biological refinement rules.

    These labels are provisional and should be checked against formal
    differential expression results in Milestone 3.
    """
    rows = []

    for cluster_id, row in cluster_scores.iterrows():
        sorted_scores = row.sort_values(ascending=False)

        top_signature = sorted_scores.index[0]
        second_signature = sorted_scores.index[1]

        top_score = sorted_scores.iloc[0]
        second_score = sorted_scores.iloc[1]
        margin = top_score - second_score

        label = LABEL_DISPLAY_NAMES[top_signature]

        # Proliferation often overlaps with tumor epithelial regions.
        if (
            top_signature == "proliferation"
            and row.get("tumor_epithelial", -999) > row.median()
        ):
            label = "Proliferative tumor"

        # Tumor epithelial clusters with high proliferation are flagged.
        elif (
            top_signature == "tumor_epithelial"
            and row.get("proliferation", -999) > sorted_scores.quantile(0.75)
        ):
            label = "Tumor epithelial/proliferative"

        # Immune signatures can overlap; if myeloid signal is strong, flag it.
        elif (
            top_signature in ["pan_immune", "t_cell", "b_cell", "myeloid_macrophage"]
            and row.get("myeloid_macrophage", -999) >= row.get("pan_immune", -999)
        ):
            label = "Myeloid/immune enriched"

        # If top two signatures are close, treat the cluster as mixed.
        # The 0.05 margin is heuristic and can be tuned.
        if margin < 0.05:
            label = (
                f"Mixed: {LABEL_DISPLAY_NAMES[top_signature]} / "
                f"{LABEL_DISPLAY_NAMES[second_signature]}"
            )

        rows.append(
            {
                "cluster": cluster_id,
                "preliminary_label": label,
                "top_signature": top_signature,
                "top_score": top_score,
                "second_signature": second_signature,
                "second_score": second_score,
                "score_margin": margin,
            }
        )

    return pd.DataFrame(rows)


def add_preliminary_marker_score_labels(adata):
    """
    Score marker-gene signatures and assign preliminary cluster labels.

    Returns
    -------
    adata:
        AnnData object with marker score columns and preliminary labels.

    score_columns:
        List of new score columns added to `adata.obs`.

    cluster_scores:
        Cluster-level mean marker signature scores.

    annotations:
        Preliminary cluster annotation table.
    """
    score_columns = []

    for signature_name, genes in MARKER_SETS.items():
        available_genes = get_available_markers(adata, genes)

        print(f"\nMarker signature: {signature_name}")
        print(f"  Requested genes: {genes}")
        print(f"  Available genes: {available_genes}")

        if len(available_genes) < 2:
            print(f"  Skipping {signature_name}: fewer than 2 genes available.")
            continue

        score_name = f"{signature_name}_score"

        sc.tl.score_genes(
            adata,
            gene_list=available_genes,
            score_name=score_name,
            use_raw=True if adata.raw is not None else False,
        )

        score_columns.append(score_name)

    print("\nCreated marker score columns:")
    print(score_columns)

    cluster_scores = (
        adata.obs[[CLUSTER_KEY] + score_columns]
        .groupby(CLUSTER_KEY, observed=True)
        .mean()
    )

    cluster_scores.columns = [
        col.replace("_score", "") for col in cluster_scores.columns
    ]

    cluster_scores.to_csv(TABLE_DIR / "cluster_marker_signature_scores.csv")

    annotations = assign_cluster_labels(cluster_scores)
    annotations.to_csv(
        TABLE_DIR / "preliminary_cluster_annotations.csv",
        index=False,
    )

    cluster_to_label = dict(
        zip(
            annotations["cluster"].astype(str),
            annotations["preliminary_label"],
        )
    )

    adata.obs["preliminary_niche_label"] = (
        adata.obs[CLUSTER_KEY]
        .astype(str)
        .map(cluster_to_label)
        .astype("category")
    )

    cluster_to_combined_label = {
        cluster: f"{cluster}: {label}"
        for cluster, label in cluster_to_label.items()
    }

    adata.obs["cluster_preliminary_label"] = (
        adata.obs[CLUSTER_KEY]
        .astype(str)
        .map(cluster_to_combined_label)
        .astype("category")
    )

    adata.obs[
        [
            CLUSTER_KEY,
            "preliminary_niche_label",
            "cluster_preliminary_label",
        ]
        + score_columns
    ].to_csv(TABLE_DIR / "spot_marker_scores_and_labels.csv")

    return adata, score_columns, cluster_scores, annotations


def plot_preliminary_marker_score_outputs(adata, score_columns, cluster_scores) -> None:
    """Generate plots for marker scores and preliminary cluster labels."""
    if score_columns:
        sc.pl.spatial(
            adata,
            color=score_columns,
            library_id=LIBRARY_ID,
            show=False,
        )
        save_current_fig("spatial_marker_signature_scores.png")

    sc.pl.spatial(
        adata,
        color=["cluster_preliminary_label"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_clusters_with_preliminary_labels.png")

    sc.pl.umap(
        adata,
        color=["cluster_preliminary_label"],
        legend_loc="right margin",
        frameon=False,
        show=False,
    )
    save_current_fig("umap_clusters_with_preliminary_labels.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cluster_scores.values, aspect="auto")

    ax.set_xticks(range(cluster_scores.shape[1]))
    ax.set_xticklabels(cluster_scores.columns, rotation=45, ha="right")

    ax.set_yticks(range(cluster_scores.shape[0]))
    ax.set_yticklabels(cluster_scores.index)

    ax.set_xlabel("Marker signature")
    ax.set_ylabel("Leiden cluster")
    ax.set_title("Mean marker signature score by Leiden cluster")

    fig.colorbar(im, ax=ax, label="Mean signature score")
    save_current_fig("cluster_marker_signature_score_heatmap.png")


def main() -> None:
    """Run preprocessing, clustering, and preliminary marker-score labeling."""
    print(f"Loading QC AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nOriginal AnnData object:")
    print(adata)

    original_n_spots = adata.n_obs
    original_n_genes = adata.n_vars

    print("\nOriginal QC summary:")
    print(
        adata.obs[
            ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
        ].describe()
    )

    # ------------------------------------------------------------------
    # 1. Conservative QC filtering
    # ------------------------------------------------------------------
    min_genes_by_counts = 500
    min_total_counts = 1000
    max_pct_counts_mt = 15

    qc_mask = (
        (adata.obs["n_genes_by_counts"] >= min_genes_by_counts)
        & (adata.obs["total_counts"] >= min_total_counts)
        & (adata.obs["pct_counts_mt"] < max_pct_counts_mt)
    )

    adata = adata[qc_mask].copy()

    filtered_n_spots = adata.n_obs

    print("\nQC filtering thresholds:")
    print(f"  n_genes_by_counts >= {min_genes_by_counts}")
    print(f"  total_counts >= {min_total_counts}")
    print(f"  pct_counts_mt < {max_pct_counts_mt}")

    print("\nFiltering summary:")
    print(f"  Original spots: {original_n_spots}")
    print(f"  Retained spots: {filtered_n_spots}")
    print(f"  Removed spots: {original_n_spots - filtered_n_spots}")
    print(f"  Original genes: {original_n_genes}")

    filtering_summary = pd.DataFrame(
        {
            "metric": [
                "original_spots",
                "retained_spots",
                "removed_spots",
                "original_genes",
                "min_genes_by_counts",
                "min_total_counts",
                "max_pct_counts_mt",
            ],
            "value": [
                original_n_spots,
                filtered_n_spots,
                original_n_spots - filtered_n_spots,
                original_n_genes,
                min_genes_by_counts,
                min_total_counts,
                max_pct_counts_mt,
            ],
        }
    )

    filtering_summary.to_csv(
        TABLE_DIR / "qc_filtering_summary.csv",
        index=False,
    )

    sc.pl.spatial(
        adata,
        color=["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_qc_after_filtering.png")

    # ------------------------------------------------------------------
    # 2. Preserve raw counts
    # ------------------------------------------------------------------
    adata.layers["counts"] = adata.X.copy()

    # ------------------------------------------------------------------
    # 3. Normalize and log-transform
    # ------------------------------------------------------------------
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store full normalized/log-transformed expression before HVG subsetting.
    adata.raw = adata

    # ------------------------------------------------------------------
    # 4. Highly variable gene selection
    # ------------------------------------------------------------------
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=3000,
        flavor="seurat",
    )

    n_hvgs = int(adata.var["highly_variable"].sum())
    print(f"\nHighly variable genes selected: {n_hvgs}")

    hvgs = adata.var[adata.var["highly_variable"]].copy()
    hvgs.to_csv(TABLE_DIR / "highly_variable_genes.csv")

    sc.pl.highly_variable_genes(adata, show=False)
    save_current_fig("highly_variable_genes.png")

    # Subset to HVGs for dimensionality reduction and clustering.
    adata = adata[:, adata.var["highly_variable"]].copy()

    # ------------------------------------------------------------------
    # 5. Scaling, PCA, neighbors, UMAP, and Leiden clustering
    # ------------------------------------------------------------------
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(
        adata,
        n_comps=50,
        svd_solver="arpack",
    )

    sc.pl.pca_variance_ratio(
        adata,
        n_pcs=50,
        log=True,
        show=False,
    )
    save_current_fig("pca_variance_ratio.png")

    sc.pp.neighbors(
        adata,
        n_neighbors=12,
        n_pcs=30,
    )

    sc.tl.umap(adata, random_state=42)

    sc.tl.leiden(
        adata,
        resolution=0.6,
        key_added=CLUSTER_KEY,
        random_state=42,
    )

    print("\nLeiden cluster counts:")
    print(adata.obs[CLUSTER_KEY].value_counts().sort_index())

    cluster_counts = (
        adata.obs[CLUSTER_KEY]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="n_spots")
    )
    cluster_counts.to_csv(TABLE_DIR / "leiden_r06_cluster_counts.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Core clustering plots
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata,
        color=[CLUSTER_KEY],
        legend_loc="on data",
        frameon=False,
        show=False,
    )
    save_current_fig("umap_leiden_r06.png")

    sc.pl.spatial(
        adata,
        color=[CLUSTER_KEY],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_leiden_r06.png")

    marker_genes = [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "COL1A1",
        "COL1A2",
        "DCN",
        "LUM",
        "PTPRC",
        "CD3D",
        "CD8A",
        "MS4A1",
        "LST1",
        "C1QA",
        "CD68",
        "MKI67",
        "TOP2A",
        "PECAM1",
        "VWF",
    ]

    available_markers = [
        gene for gene in marker_genes if gene in adata.raw.var_names
    ]

    print("\nAvailable marker genes for plotting:")
    print(available_markers)

    if available_markers:
        sc.pl.spatial(
            adata,
            color=available_markers,
            library_id=LIBRARY_ID,
            use_raw=True,
            show=False,
        )
        save_current_fig("spatial_marker_genes_normalized.png")

    # ------------------------------------------------------------------
    # 7. Preliminary marker-score annotation
    # ------------------------------------------------------------------
    adata, score_columns, cluster_scores, annotations = (
        add_preliminary_marker_score_labels(adata)
    )

    print("\nPreliminary cluster annotations:")
    print(annotations)

    plot_preliminary_marker_score_outputs(
        adata=adata,
        score_columns=score_columns,
        cluster_scores=cluster_scores,
    )

    # ------------------------------------------------------------------
    # 8. Save processed object
    # ------------------------------------------------------------------
    output_h5ad = PROCESSED_DIR / "visium_human_breast_cancer_processed_clustered.h5ad"
    adata.write_h5ad(output_h5ad)

    print(f"\nSaved processed AnnData object to: {output_h5ad}")
    print("\nMilestone 2 complete.")


if __name__ == "__main__":
    main()
