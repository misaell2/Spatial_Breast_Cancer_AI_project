from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_processed_clustered.h5ad"
)

OUTPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_marker_annotated.h5ad"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "03_marker_gene_analysis"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LIBRARY_ID = "Visium_Human_Breast_Cancer"
CLUSTER_KEY = "leiden_r06"


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
    """Save current matplotlib figure and close it."""
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def get_available_genes(adata, genes):
    """Return genes available in adata.raw if present; otherwise adata.var."""
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
    """Return filtered top markers per cluster."""
    df = marker_df.copy()

    # Scanpy's rank_genes_groups dataframe usually has:
    # group, names, scores, logfoldchanges, pvals, pvals_adj
    df = df.rename(columns={"names": "gene"})

    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"] >= min_logfc]

    if "pvals_adj" in df.columns:
        df = df[df["pvals_adj"] <= max_adj_pval]

    top_rows = []
    for cluster, sub_df in df.groupby("group"):
        top_rows.append(sub_df.head(top_n))

    if not top_rows:
        return pd.DataFrame()

    return pd.concat(top_rows, axis=0).reset_index(drop=True)


def suggest_cluster_labels(top_markers_df: pd.DataFrame, cluster_counts: pd.Series) -> pd.DataFrame:
    """
    Suggest biological labels based on overlap between top DE markers and known marker sets.

    These are first-pass suggestions. They should be inspected manually.
    """
    annotation_rows = []

    for cluster in sorted(top_markers_df["group"].astype(str).unique(), key=lambda x: int(x)):
        cluster_markers = (
            top_markers_df[top_markers_df["group"].astype(str) == cluster]["gene"]
            .head(50)
            .tolist()
        )
        cluster_marker_set = set(cluster_markers)

        scores = {}
        matched_markers = {}

        for label, markers in KNOWN_MARKER_SETS.items():
            overlap = sorted(cluster_marker_set.intersection(set(markers)))
            scores[label] = len(overlap)
            matched_markers[label] = overlap

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        best_matches = matched_markers[best_label]

        if best_score == 0:
            suggested_label = "Uncertain/mixed"
        else:
            suggested_label = best_label

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


def add_manual_or_suggested_labels(adata, annotation_df: pd.DataFrame):
    """
    Add suggested labels to adata.obs.

    Later, you can manually fill in the manual_label column and rerun
    this logic to use your final biological labels.
    """
    label_map = {}

    for _, row in annotation_df.iterrows():
        cluster = str(row["cluster"])

        if isinstance(row.get("manual_label", ""), str) and row["manual_label"].strip():
            label = row["manual_label"].strip()
        else:
            label = row["suggested_label"]

        label_map[cluster] = label

    combined_label_map = {
        cluster: f"{cluster}: {label}" for cluster, label in label_map.items()
    }

    adata.obs["suggested_niche_label"] = (
        adata.obs[CLUSTER_KEY].astype(str).map(label_map).astype("category")
    )

    adata.obs["cluster_suggested_niche_label"] = (
        adata.obs[CLUSTER_KEY].astype(str).map(combined_label_map).astype("category")
    )

    return adata


def main():
    print(f"Loading clustered AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded AnnData object:")
    print(adata)

    if CLUSTER_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find {CLUSTER_KEY} in adata.obs")

    print("\nCluster counts:")
    cluster_counts = adata.obs[CLUSTER_KEY].astype(str).value_counts().sort_index()
    print(cluster_counts)

    # ------------------------------------------------------------------
    # 1. Run marker gene discovery
    # ------------------------------------------------------------------
    print("\nRunning differential expression with Wilcoxon rank-sum test...")

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
    marker_df = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        key="rank_genes_leiden_r06",
    )

    marker_df = marker_df.rename(columns={"names": "gene"})

    marker_table_path = TABLE_DIR / "leiden_r06_full_marker_gene_table.csv"
    marker_df.to_csv(marker_table_path, index=False)

    print(f"\nSaved full marker gene table to: {marker_table_path}")

    # ------------------------------------------------------------------
    # 3. Export filtered top markers
    # ------------------------------------------------------------------
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
    annotation_df = suggest_cluster_labels(top_markers_df, cluster_counts)

    annotation_path = TABLE_DIR / "leiden_r06_cluster_annotation_template.csv"
    annotation_df.to_csv(annotation_path, index=False)

    print(f"\nSaved cluster annotation template to: {annotation_path}")
    print("\nPreliminary annotation table:")
    print(annotation_df)

    # Add suggested labels back into AnnData.
    adata = add_manual_or_suggested_labels(adata, annotation_df)

    # ------------------------------------------------------------------
    # 5. Plot ranked marker genes
    # ------------------------------------------------------------------
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
    plot_genes = []

    for cluster, sub_df in top_markers_df.groupby("group"):
        plot_genes.extend(sub_df["gene"].head(3).tolist())

    # Keep unique genes while preserving order.
    plot_genes = list(dict.fromkeys(plot_genes))

    # Avoid making unreadable plots.
    plot_genes = plot_genes[:40]

    print("\nTop marker genes used for dotplot/matrixplot:")
    print(plot_genes)

    if plot_genes:
        sc.pl.dotplot(
            adata,
            var_names=plot_genes,
            groupby=CLUSTER_KEY,
            use_raw=True if adata.raw is not None else False,
            standard_scale="var",
            show=False,
        )
        save_current_fig("dotplot_top_cluster_markers.png")

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
    known_marker_genes = []
    for genes in KNOWN_MARKER_SETS.values():
        known_marker_genes.extend(genes)

    known_marker_genes = list(dict.fromkeys(known_marker_genes))
    known_marker_genes = get_available_genes(adata, known_marker_genes)

    # Keep this plot readable.
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
    sc.pl.spatial(
        adata,
        color=["cluster_suggested_niche_label"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_suggested_niche_labels.png")

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
    adata.write_h5ad(OUTPUT_H5AD)

    print(f"\nSaved marker-annotated AnnData object to: {OUTPUT_H5AD}")
    print("\nMilestone 3 complete.")


if __name__ == "__main__":
    main()
