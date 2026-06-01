"""
Milestone 3: Marker gene discovery and preliminary biological annotation.

Purpose
-------
This script performs formal marker gene analysis for each Leiden cluster
identified during preprocessing and clustering.

The previous script, `02_preprocess_cluster.py`, performs QC, normalization,
dimensionality reduction, Leiden clustering, and preliminary marker-score
labeling. This script goes one step further by asking:

    Which genes are statistically enriched in each Leiden cluster?

The output marker tables are then used to manually interpret spatial tissue
niches such as:

    - tumor epithelial
    - luminal-like tumor
    - myeloid/APC-enriched immune
    - B-cell/plasma-cell immune
    - adipocyte/fat-associated
    - hypoxic/metabolic tumor-like
    - mixed or uncertain regions

Input
-----
    data/processed/visium_human_breast_cancer_processed_clustered.h5ad

Expected contents of the input AnnData object:
    - Leiden cluster labels in `adata.obs["leiden_r06"]`
    - normalized/log-transformed expression in `adata.raw`
    - UMAP coordinates in `adata.obsm["X_umap"]`
    - spatial coordinates in `adata.obsm["spatial"]`
    - Visium image metadata in `adata.uns["spatial"]`

Outputs
-------
AnnData:

    data/processed/visium_human_breast_cancer_marker_annotated.h5ad

Tables:

    results/tables/leiden_r06_full_marker_gene_table.csv
    results/tables/leiden_r06_top20_marker_genes_per_cluster.csv
    results/tables/leiden_r06_cluster_annotation_template.csv

Figures:

    results/figures/03_marker_gene_analysis/

Main analysis steps
-------------------
1. Load the clustered AnnData object.
2. Run differential expression for each Leiden cluster.
3. Export a full marker gene table.
4. Filter and export top marker genes per cluster.
5. Suggest first-pass labels based on known marker overlap.
6. Generate marker gene plots.
7. Save an AnnData object with suggested labels.

Important notes
---------------
The automated labels created here are only first-pass suggestions. The final
biological labels are assigned manually in:

    data_manifest/annotations/leiden_r06_manual_cluster_annotations.csv

and then applied in:

    src/analysis/04_apply_manual_annotations.py

Alternative analysis options
----------------------------
Differential expression methods:
    - method="wilcoxon"  : robust nonparametric test; used here
    - method="t-test"    : faster, but assumptions may be less appropriate
    - method="t-test_overestim_var" : conservative t-test option in Scanpy
    - method="logreg"    : logistic regression-based marker selection

Comparison reference:
    - reference="rest"   : each cluster vs all other clusters; Scanpy default
    - reference="<cluster_id>" : compare each cluster to one chosen reference

Marker filtering:
    - min_logfc could be stricter, e.g. 0.5 or 1.0
    - max_adj_pval could be stricter, e.g. 0.01
    - markers could also be filtered by pct_nz_group and pct_nz_reference

Biological interpretation:
    - marker overlap is useful but incomplete
    - manual review of top genes, dotplots, and spatial plots is required
    - pathway enrichment could be added later using g:Profiler, Enrichr,
      decoupler, GSEA, or clusterProfiler in R
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This script lives in:
#   src/analysis/03_marker_gene_analysis.py
#
# parents[2] moves from:
#   src/analysis/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input from Milestone 2.
# This object should contain the processed expression matrix and Leiden labels.
INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_processed_clustered.h5ad"
)

# Output object with differential expression results and suggested labels.
OUTPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_marker_annotated.h5ad"
)

# Output directories for figures and tables.
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "03_marker_gene_analysis"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

# Create output directories if they do not exist.
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset and clustering metadata
# ---------------------------------------------------------------------
# This should match the key in adata.uns["spatial"].
LIBRARY_ID = "Visium_Human_Breast_Cancer"

# Leiden cluster key created in Milestone 2.
# If you later test other Leiden resolutions, update this key or make it a
# command-line argument.
CLUSTER_KEY = "leiden_r06"


# ---------------------------------------------------------------------
# Known biological marker sets
# ---------------------------------------------------------------------
# These marker sets are used to help suggest labels for each Leiden cluster.
# The formal labels are still reviewed manually later.
#
# Notes:
#   - These are simple, interpretable marker sets.
#   - They are not meant to be exhaustive.
#   - Visium spots can contain mixed cell populations.
#   - Some markers are cell-type-specific, while others reflect biological
#     programs such as proliferation or hypoxia/stress.
#
# Possible future improvements:
#   - Add breast cancer subtype markers.
#   - Add CAF subtype markers.
#   - Split macrophage/myeloid markers into finer programs.
#   - Add cytotoxic T/NK markers as a separate signature.
#   - Use curated databases such as MSigDB, CellMarker, PanglaoDB, or Azimuth.
KNOWN_MARKER_SETS = {
    "Tumor epithelial": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "KRT7",
        "MUC1",
        "TACSTD2",
    ],
    "Stromal/CAF": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "ACTA2",
        "TAGLN",
        "FAP",
        "PDGFRA",
    ],
    "Pan-immune": [
        "PTPRC",
        "LCP1",
        "CD52",
        "CORO1A",
        "CXCL13",
    ],
    "T-cell enriched": [
        "CD3D",
        "CD3E",
        "CD2",
        "TRAC",
        "CD8A",
        "CD8B",
        "GZMB",
        "NKG7",
    ],
    "B-cell enriched": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
        "MZB1",
        "JCHAIN",
    ],
    "Myeloid/macrophage": [
        "LST1",
        "C1QA",
        "C1QB",
        "C1QC",
        "CD68",
        "TYROBP",
        "FCER1G",
        "AIF1",
    ],
    "Endothelial/vascular": [
        "PECAM1",
        "VWF",
        "KDR",
        "ENG",
        "PLVAP",
        "CLDN5",
    ],
    "Proliferative": [
        "MKI67",
        "TOP2A",
        "PCNA",
        "MCM5",
        "STMN1",
        "UBE2C",
    ],
    "Adipocyte/fat-associated": [
        "ADIPOQ",
        "PLIN1",
        "FABP4",
        "LPL",
        "APOE",
    ],
    "Hypoxia/stress": [
        "VEGFA",
        "CA9",
        "LDHA",
        "ENO1",
        "SLC2A1",
    ],
}


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.

    Parameters
    ----------
    filename:
        Output filename inside FIGURE_DIR.

    Why close figures?
    ------------------
    Closing after saving prevents later plots from overlapping and avoids
    excessive memory usage when many plots are created in one script.
    """
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def get_available_genes(adata, genes: list[str]) -> list[str]:
    """
    Return genes that are available in the AnnData object.

    Parameters
    ----------
    adata:
        AnnData object containing gene expression.

    genes:
        Candidate gene symbols to check.

    Returns
    -------
    list[str]
        Genes present in either `adata.raw.var_names` or `adata.var_names`.

    Why use adata.raw?
    ------------------
    The processed object is subset to highly variable genes for PCA and
    clustering, but `adata.raw` stores the full normalized/log-transformed
    expression matrix before HVG subsetting. Marker genes may not be HVGs, so
    checking `adata.raw` preserves access to more biologically relevant genes.

    Alternative
    -----------
    If using Ensembl IDs, add a gene-symbol-to-Ensembl mapping step before
    checking marker availability.
    """
    if adata.raw is not None:
        available = set(adata.raw.var_names)
    else:
        available = set(adata.var_names)

    return [gene for gene in genes if gene in available]


def extract_top_markers(
    marker_df: pd.DataFrame,
    top_n: int = 20,
    min_logfc: float = 0.25,
    max_adj_pval: float = 0.05,
) -> pd.DataFrame:
    """
    Filter the full marker table and return top markers per cluster.

    Parameters
    ----------
    marker_df:
        DataFrame returned by Scanpy's rank_genes_groups results.

    top_n:
        Number of top markers to keep per cluster after filtering.

    min_logfc:
        Minimum log fold-change threshold. Larger values are more stringent.

    max_adj_pval:
        Maximum adjusted p-value threshold.

    Returns
    -------
    pd.DataFrame
        Filtered marker table with up to `top_n` markers per cluster.

    Notes on thresholds
    -------------------
    Current choices:
        min_logfc = 0.25
        max_adj_pval = 0.05

    These are intentionally permissive because this is an exploratory spatial
    transcriptomics workflow.

    Alternative thresholds:
        - min_logfc = 0.5 for more clearly enriched markers
        - min_logfc = 1.0 for very strong markers
        - max_adj_pval = 0.01 for stricter statistical significance

    Additional useful filters:
        - require pct_nz_group > 0.10 or 0.25
        - require pct_nz_group - pct_nz_reference > 0.10
        - remove mitochondrial or ribosomal genes from annotation candidates
    """
    df = marker_df.copy()

    # Scanpy marker tables may store gene names in a column called "names".
    # This project standardizes that column to "gene".
    df = df.rename(columns={"names": "gene"})

    # Keep only genes with sufficient log fold-change if that column exists.
    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"] >= min_logfc]

    # Keep only genes with significant adjusted p-values if that column exists.
    if "pvals_adj" in df.columns:
        df = df[df["pvals_adj"] <= max_adj_pval]

    # Keep the first `top_n` rows per cluster.
    # Scanpy returns genes already ranked by the chosen test statistic.
    top_rows = []
    for cluster, sub_df in df.groupby("group"):
        top_rows.append(sub_df.head(top_n))

    # Return an empty DataFrame if filtering removed all markers.
    if not top_rows:
        return pd.DataFrame()

    return pd.concat(top_rows, axis=0).reset_index(drop=True)


def suggest_cluster_labels(
    top_markers_df: pd.DataFrame,
    cluster_counts: pd.Series,
) -> pd.DataFrame:
    """
    Suggest biological labels using overlap with known marker sets.

    Parameters
    ----------
    top_markers_df:
        Filtered marker table containing top marker genes per cluster.

    cluster_counts:
        Series mapping cluster ID to number of spots.

    Returns
    -------
    pd.DataFrame
        Annotation template with suggested labels, matched markers, top DE
        markers, and blank columns for manual labels and notes.

    Labeling strategy
    -----------------
    For each cluster:
        1. Take the top DE markers.
        2. Compare them to known marker sets.
        3. Count overlaps with each known marker set.
        4. Suggest the label with the highest overlap.

    Limitations
    -----------
    Marker overlap is a simple heuristic. It can miss biologically meaningful
    clusters if:
        - cluster markers are not in the known marker list
        - a cluster reflects a pathway/state rather than a cell type
        - the spot contains mixed cell populations
        - markers are shared between related cell types

    Alternative approaches
    ----------------------
    - score marker sets per cluster using average signature scores
    - use pathway enrichment on cluster marker genes
    - use a single-cell reference for cell-type deconvolution
    - combine marker overlap with manual review of spatial plots
    """
    annotation_rows = []

    # Sort clusters numerically so output table is easy to read.
    for cluster in sorted(
        top_markers_df["group"].astype(str).unique(),
        key=lambda x: int(x),
    ):
        # Extract up to the top 50 marker genes for this cluster.
        # The top 10 are reported separately, but the top 50 are used for
        # marker-set overlap to avoid being too sensitive to only a few genes.
        cluster_markers = (
            top_markers_df[top_markers_df["group"].astype(str) == cluster]["gene"]
            .head(50)
            .tolist()
        )
        cluster_marker_set = set(cluster_markers)

        # Store overlap counts and matched markers for each known marker set.
        scores = {}
        matched_markers = {}

        for label, markers in KNOWN_MARKER_SETS.items():
            overlap = sorted(cluster_marker_set.intersection(set(markers)))
            scores[label] = len(overlap)
            matched_markers[label] = overlap

        # Pick the marker set with the most overlap.
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        best_matches = matched_markers[best_label]

        # If no known markers overlap, leave the cluster as uncertain.
        if best_score == 0:
            suggested_label = "Uncertain/mixed"
        else:
            suggested_label = best_label

        # Save the top 10 DE markers to make manual review easier.
        top_10_markers = cluster_markers[:10]

        annotation_rows.append(
            {
                "cluster": cluster,
                "n_spots": int(cluster_counts.loc[cluster]),
                "suggested_label": suggested_label,
                "marker_set_overlap_count": best_score,
                "matched_known_markers": ";".join(best_matches),
                "top_10_de_markers": ";".join(top_10_markers),
                "manual_label": "",
                "notes": "",
            }
        )

    return pd.DataFrame(annotation_rows)


def add_manual_or_suggested_labels(
    adata,
    annotation_df: pd.DataFrame,
):
    """
    Add suggested or manually edited labels to `adata.obs`.

    Parameters
    ----------
    adata:
        AnnData object with Leiden cluster labels.

    annotation_df:
        Annotation table containing at least:
            - cluster
            - suggested_label
            - manual_label

    Returns
    -------
    adata:
        AnnData object with added label columns.

    Behavior
    --------
    If `manual_label` is filled in, it is used.
    Otherwise, the script falls back to `suggested_label`.

    Why this is useful
    ------------------
    This allows the same function to work before and after manual curation.

    In this project, final manual labels are handled more formally in:
        src/analysis/04_apply_manual_annotations.py
    """
    label_map = {}

    for _, row in annotation_df.iterrows():
        cluster = str(row["cluster"])

        # Prefer a manually curated label if one exists.
        if isinstance(row.get("manual_label", ""), str) and row["manual_label"].strip():
            label = row["manual_label"].strip()
        else:
            label = row["suggested_label"]

        label_map[cluster] = label

    # Combined labels preserve both cluster number and biological interpretation.
    combined_label_map = {
        cluster: f"{cluster}: {label}" for cluster, label in label_map.items()
    }

    # Add biological niche label without cluster number.
    adata.obs["suggested_niche_label"] = (
        adata.obs[CLUSTER_KEY].astype(str).map(label_map).astype("category")
    )

    # Add combined cluster + label for easier figure legends.
    adata.obs["cluster_suggested_niche_label"] = (
        adata.obs[CLUSTER_KEY].astype(str).map(combined_label_map).astype("category")
    )

    return adata


def main() -> None:
    """
    Run formal marker gene analysis and create annotation outputs.

    Workflow
    --------
    1. Load clustered AnnData object.
    2. Run differential expression using Scanpy.
    3. Export full and filtered marker gene tables.
    4. Create an annotation template for manual review.
    5. Add suggested labels to AnnData.
    6. Generate marker plots and spatial label plots.
    7. Save marker-annotated AnnData object.
    """
    print(f"Loading clustered AnnData object from: {INPUT_H5AD}")

    # Load the processed clustered AnnData object.
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded AnnData object:")
    print(adata)

    # Confirm the expected Leiden cluster column exists.
    if CLUSTER_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find {CLUSTER_KEY} in adata.obs")

    print("\nCluster counts:")

    # Cluster counts help evaluate label imbalance and cluster size.
    # Small clusters may be biologically interesting, but they can also reflect
    # edge effects, technical variation, or rare mixed regions.
    cluster_counts = adata.obs[CLUSTER_KEY].astype(str).value_counts().sort_index()
    print(cluster_counts)

    # ------------------------------------------------------------------
    # 1. Run marker gene discovery
    # ------------------------------------------------------------------
    print("\nRunning differential expression with Wilcoxon rank-sum test...")

    # `sc.tl.rank_genes_groups` compares each cluster against a reference group.
    # By default, Scanpy uses reference="rest", meaning each cluster is compared
    # against all other spots.
    #
    # Current settings:
    #   groupby=CLUSTER_KEY
    #       Find markers for each Leiden cluster.
    #
    #   method="wilcoxon"
    #       Nonparametric Wilcoxon rank-sum test. This is commonly used for
    #       single-cell/spatial marker discovery because it does not assume
    #       normally distributed expression.
    #
    #   use_raw=True if available
    #       Use the full normalized/log-transformed gene matrix stored in
    #       adata.raw, rather than only highly variable genes.
    #
    #   pts=True
    #       Calculate the fraction of spots expressing each gene in the group
    #       and reference. These columns help judge whether a marker is broadly
    #       present or driven by only a few spots.
    #
    # Alternative useful parameters:
    #   method="t-test"
    #   method="t-test_overestim_var"
    #   method="logreg"
    #   reference="rest"
    #   reference="0"  # compare all clusters to cluster 0
    #   corr_method="benjamini-hochberg"
    #   n_genes=100
    #
    # Potential future improvement:
    #   Exclude mitochondrial/ribosomal genes from candidate markers before
    #   biological interpretation.
    sc.tl.rank_genes_groups(
        adata,
        groupby=CLUSTER_KEY,
        method="wilcoxon",
        use_raw=True if adata.raw is not None else False,
        pts=True,
        key_added="rank_genes_leiden_r06",
    )

    print("Differential expression complete.")

    # ------------------------------------------------------------------
    # 2. Export full marker table
    # ------------------------------------------------------------------
    # Convert Scanpy's internal rank_genes_groups result into a long DataFrame.
    marker_df = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        key="rank_genes_leiden_r06",
    )

    # Standardize gene column name.
    marker_df = marker_df.rename(columns={"names": "gene"})

    # Save the full table for transparency and downstream review.
    marker_table_path = TABLE_DIR / "leiden_r06_full_marker_gene_table.csv"
    marker_df.to_csv(marker_table_path, index=False)

    print(f"\nSaved full marker gene table to: {marker_table_path}")

    # ------------------------------------------------------------------
    # 3. Export filtered top markers
    # ------------------------------------------------------------------
    # Keep a smaller marker table that is easier to inspect manually.
    #
    # Current thresholds:
    #   top_n=20
    #   min_logfc=0.25
    #   max_adj_pval=0.05
    #
    # Alternative:
    #   Use min_logfc=0.5 for more stringent marker genes.
    #   Use top_n=50 if you plan to run pathway enrichment.
    top_markers_df = extract_top_markers(
        marker_df.rename(columns={"gene": "names"}),
        top_n=20,
        min_logfc=0.25,
        max_adj_pval=0.05,
    )

    top_marker_path = TABLE_DIR / "leiden_r06_top20_marker_genes_per_cluster.csv"
    top_markers_df.to_csv(top_marker_path, index=False)

    print(f"Saved top marker genes to: {top_marker_path}")

    print("\nTop 5 markers per cluster:")
    for cluster, sub_df in top_markers_df.groupby("group"):
        genes = sub_df["gene"].head(5).tolist()
        print(f"  Cluster {cluster}: {', '.join(genes)}")

    # ------------------------------------------------------------------
    # 4. Create preliminary annotation table
    # ------------------------------------------------------------------
    # This table is designed for manual editing.
    # The `manual_label` and `notes` columns are intentionally blank.
    annotation_df = suggest_cluster_labels(top_markers_df, cluster_counts)

    annotation_path = TABLE_DIR / "leiden_r06_cluster_annotation_template.csv"
    annotation_df.to_csv(annotation_path, index=False)

    print(f"\nSaved cluster annotation template to: {annotation_path}")
    print("\nPreliminary annotation table:")
    print(annotation_df)

    # Add suggested labels back into AnnData for immediate plotting.
    # Final labels are applied later after manual annotation review.
    adata = add_manual_or_suggested_labels(adata, annotation_df)

    # ------------------------------------------------------------------
    # 5. Plot ranked marker genes
    # ------------------------------------------------------------------
    # This plot shows the top ranked genes for each cluster directly from the
    # differential expression test.
    #
    # Useful parameters:
    #   n_genes=5   gives a compact summary
    #   n_genes=10  gives more context but can be crowded
    #   sharey=False improves readability across clusters
    sc.pl.rank_genes_groups(
        adata,
        key="rank_genes_leiden_r06",
        n_genes=5,
        sharey=False,
        show=False,
    )
    save_current_fig("rank_genes_groups_top5_per_cluster.png")

    # ------------------------------------------------------------------
    # 6. Dotplot and matrixplot of top markers
    # ------------------------------------------------------------------
    # Select the top 3 marker genes per cluster for summary plots.
    # This gives a compact set of genes representing all clusters.
    plot_genes = []

    for cluster, sub_df in top_markers_df.groupby("group"):
        plot_genes.extend(sub_df["gene"].head(3).tolist())

    # Remove duplicates while preserving order.
    plot_genes = list(dict.fromkeys(plot_genes))

    # Limit to 40 genes to avoid unreadable figures.
    # Alternative:
    #   Increase to 60-80 for exploratory internal plots.
    #   Keep <=40 for GitHub/README-friendly figures.
    plot_genes = plot_genes[:40]

    print("\nTop marker genes used for dotplot/matrixplot:")
    print(plot_genes)

    if plot_genes:
        # Dotplot shows both:
        #   - average expression
        #   - fraction of spots expressing each gene
        #
        # `standard_scale="var"` scales each gene across clusters, making it
        # easier to see which cluster has relatively high expression.
        #
        # Alternative:
        #   standard_scale=None preserves raw normalized expression scale.
        sc.pl.dotplot(
            adata,
            var_names=plot_genes,
            groupby=CLUSTER_KEY,
            use_raw=True if adata.raw is not None else False,
            standard_scale="var",
            show=False,
        )
        save_current_fig("dotplot_top_cluster_markers.png")

        # Matrixplot shows expression intensity across clusters in a compact
        # heatmap-like format.
        sc.pl.matrixplot(
            adata,
            var_names=plot_genes,
            groupby=CLUSTER_KEY,
            use_raw=True if adata.raw is not None else False,
            standard_scale="var",
            show=False,
        )
        save_current_fig("matrixplot_top_cluster_markers.png")

    # ------------------------------------------------------------------
    # 7. Plot known biological marker genes
    # ------------------------------------------------------------------
    # This plot checks known biological markers across all clusters.
    # It complements the unbiased DE marker table.
    known_marker_genes = []

    for genes in KNOWN_MARKER_SETS.values():
        known_marker_genes.extend(genes)

    # Remove duplicate genes while preserving order.
    known_marker_genes = list(dict.fromkeys(known_marker_genes))

    # Keep only genes available in the dataset.
    known_marker_genes = get_available_genes(adata, known_marker_genes)

    # Keep this plot readable.
    # Alternative:
    #   Split into several dotplots by biological category if the marker list
    #   becomes too large.
    known_marker_genes_for_plot = known_marker_genes[:50]

    if known_marker_genes_for_plot:
        sc.pl.dotplot(
            adata,
            var_names=known_marker_genes_for_plot,
            groupby=CLUSTER_KEY,
            use_raw=True if adata.raw is not None else False,
            standard_scale="var",
            show=False,
        )
        save_current_fig("dotplot_known_biological_markers.png")

    # ------------------------------------------------------------------
    # 8. Plot suggested labels spatially and on UMAP
    # ------------------------------------------------------------------
    # Spatial plot:
    #   Checks whether suggested labels occupy coherent tissue regions.
    sc.pl.spatial(
        adata,
        color=["cluster_suggested_niche_label"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_suggested_niche_labels.png")

    # UMAP plot:
    #   Checks whether suggested labels separate in expression space.
    sc.pl.umap(
        adata,
        color=["cluster_suggested_niche_label"],
        legend_loc="right margin",
        frameon=False,
        show=False,
    )
    save_current_fig("umap_suggested_niche_labels.png")

    # ------------------------------------------------------------------
    # 9. Plot selected key spatial markers
    # ------------------------------------------------------------------
    # These genes were chosen to represent major biological programs:
    #   epithelial/tumor, stromal, immune, myeloid, proliferation, endothelial.
    selected_spatial_markers = [
        "EPCAM",
        "KRT18",
        "KRT19",
        "COL1A1",
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

    selected_spatial_markers = get_available_genes(adata, selected_spatial_markers)

    if selected_spatial_markers:
        sc.pl.spatial(
            adata,
            color=selected_spatial_markers,
            library_id=LIBRARY_ID,
            use_raw=True if adata.raw is not None else False,
            show=False,
        )
        save_current_fig("spatial_selected_biological_markers.png")

    # ------------------------------------------------------------------
    # 10. Save updated AnnData object
    # ------------------------------------------------------------------
    # This object keeps:
    #   - the processed expression object
    #   - DE results in adata.uns["rank_genes_leiden_r06"]
    #   - suggested niche labels in adata.obs
    adata.write_h5ad(OUTPUT_H5AD)

    print(f"\nSaved marker-annotated AnnData object to: {OUTPUT_H5AD}")
    print("\nMilestone 3 complete.")


if __name__ == "__main__":
    main()
