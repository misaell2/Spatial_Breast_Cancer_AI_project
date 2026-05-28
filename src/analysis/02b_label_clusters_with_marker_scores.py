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
    / "visium_human_breast_cancer_clustered_labeled.h5ad"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "02b_marker_score_annotation"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LIBRARY_ID = "Visium_Human_Breast_Cancer"
CLUSTER_KEY = "leiden_r06"


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


def save_current_fig(filename: str):
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def get_available_markers(adata, genes):
    """Return marker genes available in adata.raw if present, otherwise adata.var."""
    if adata.raw is not None:
        gene_index = set(adata.raw.var_names)
    else:
        gene_index = set(adata.var_names)

    return [gene for gene in genes if gene in gene_index]


def assign_cluster_labels(cluster_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a preliminary label to each Leiden cluster from marker scores.

    This is intentionally conservative. It gives a first-pass biological
    interpretation, but these labels should be refined in the formal
    marker-gene analysis step.
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

        # Helpful compound rules.
        # Proliferation often overlaps tumor regions, so label it as
        # proliferative tumor if epithelial signal is also high.
        if (
            top_signature == "proliferation"
            and row.get("tumor_epithelial", -999) > row.median()
        ):
            label = "Proliferative tumor"

        elif (
            top_signature == "tumor_epithelial"
            and row.get("proliferation", -999) > sorted_scores.quantile(0.75)
        ):
            label = "Tumor epithelial/proliferative"

        elif (
            top_signature in ["pan_immune", "t_cell", "b_cell", "myeloid_macrophage"]
            and row.get("myeloid_macrophage", -999) >= row.get("pan_immune", -999)
        ):
            label = "Myeloid/immune enriched"

        # If the top two scores are very close, flag the cluster as mixed.
        # You can tune this after viewing the tables.
        if margin < 0.05:
            label = f"Mixed: {LABEL_DISPLAY_NAMES[top_signature]} / {LABEL_DISPLAY_NAMES[second_signature]}"

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


def main():
    print(f"Loading clustered AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded object:")
    print(adata)

    if CLUSTER_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find cluster key in adata.obs: {CLUSTER_KEY}")

    # ------------------------------------------------------------------
    # 1. Score each biological marker set per spot
    # ------------------------------------------------------------------
    score_columns = []

    for signature_name, genes in MARKER_SETS.items():
        available_genes = get_available_markers(adata, genes)

        print(f"\n{signature_name}")
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

    print("\nCreated score columns:")
    print(score_columns)

    # ------------------------------------------------------------------
    # 2. Aggregate signature scores by Leiden cluster
    # ------------------------------------------------------------------
    cluster_scores = (
        adata.obs[[CLUSTER_KEY] + score_columns]
        .groupby(CLUSTER_KEY, observed=True)
        .mean()
    )

    # Make column names cleaner for tables.
    cluster_scores.columns = [
        col.replace("_score", "") for col in cluster_scores.columns
    ]

    cluster_scores.to_csv(TABLE_DIR / "cluster_marker_signature_scores.csv")
    print("\nCluster-level marker signature scores:")
    print(cluster_scores)

    # ------------------------------------------------------------------
    # 3. Assign preliminary biological labels
    # ------------------------------------------------------------------
    annotations = assign_cluster_labels(cluster_scores)
    annotations.to_csv(
        TABLE_DIR / "preliminary_cluster_annotations.csv",
        index=False,
    )

    print("\nPreliminary cluster annotations:")
    print(annotations)

    cluster_to_label = dict(
        zip(
            annotations["cluster"].astype(str),
            annotations["preliminary_label"],
        )
    )

    # Add cluster-level labels to each spot.
    adata.obs["preliminary_niche_label"] = (
        adata.obs[CLUSTER_KEY]
        .astype(str)
        .map(cluster_to_label)
        .astype("category")
    )

    # Add combined cluster + label so plots retain cluster numbers too.
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

    # Save spot-level annotation table.
    adata.obs[
        [
            CLUSTER_KEY,
            "preliminary_niche_label",
            "cluster_preliminary_label",
        ]
        + score_columns
    ].to_csv(TABLE_DIR / "spot_marker_scores_and_labels.csv")

    # ------------------------------------------------------------------
    # 4. Plot signature scores on tissue
    # ------------------------------------------------------------------
    sc.pl.spatial(
        adata,
        color=score_columns,
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_marker_signature_scores.png")

    # ------------------------------------------------------------------
    # 5. Plot preliminary biological labels
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Plot cluster-level signature score heatmap
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 7. Save updated AnnData object
    # ------------------------------------------------------------------
    adata.write_h5ad(OUTPUT_H5AD)
    print(f"\nSaved labeled AnnData object to: {OUTPUT_H5AD}")
    print("\nMilestone 2.5 complete.")


if __name__ == "__main__":
    main()
