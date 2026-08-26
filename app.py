"""Polygenic expression-specificity tester for healthy-adult immune cells."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "immune_aging_streamlit_matplotlib"),
)

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
import streamlit as st

from analysis import (
    convert_to_human,
    load_ortholog_map,
    load_rank_matrix,
    parse_gene_list,
    run_auroc_analysis,
)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
RANK_MATRIX_PATH = DATA_DIR / "cell_type_average_expression.csv.gz"

DEFAULT_GENES = """APOE
ABCA1
ABCA7
ABI3
ACE
ADAM17
ADAMTS1
ANK3
ANKH
APH1B
APP
BCKDK
BIN1
BLNK
CASS4
CD2AP
CLNK
CLU
COX7C
CR1
CTSB
CTSH
DOC2A
EED
EPDR1
EPHA1
FERMT2
FOXF1
GRN
HLA-DQA1
HS3ST5
ICA1
IDUA
IGHG3
IGHV3-65
IL34
INPP5D
JAZF1
KLF16
LILRB2
MAF
MINDY2
MME
MS4A4A
MYO15A
NCK2
PLCG2
PLEKHA1
PRDM7
PRKD3
PTK2B
RASGEF1C
RBCK1
RHOH
SCIMP
SEC61G
SHARPIN
SIGLEC11
SLC24A4
SLC2A4RG
SNX1
SORL1
SORT1
SPDYE3
SPI1
SPPL2A
TMEM106B
TNIP1
TPCN1
TREM2
TREML2
TSPAN14
TSPOAP1
UMAD1
UNC5CL
USP6NL
WDR12
WDR81
WNT3"""


@st.cache_resource(show_spinner="Loading immune-cell rank matrix...")
def get_rank_matrix():
    return load_rank_matrix(RANK_MATRIX_PATH)


@st.cache_resource(show_spinner=False)
def get_ortholog_map():
    return load_ortholog_map(DATA_DIR / "homologene_to_human.csv.gz")


@st.cache_data(show_spinner=False)
def create_heatmap_pdf(heatmap: pd.DataFrame) -> bytes:
    clustered_heatmap = heatmap
    if heatmap.shape[0] > 1:
        row_order = leaves_list(
            linkage(heatmap.to_numpy(), method="complete", metric="euclidean")
        )
        clustered_heatmap = clustered_heatmap.iloc[row_order, :]
    if heatmap.shape[1] > 1:
        column_order = leaves_list(
            linkage(heatmap.to_numpy().T, method="complete", metric="euclidean")
        )
        clustered_heatmap = clustered_heatmap.iloc[:, column_order]

    width = min(40, max(8, 3 + heatmap.shape[1] * 0.15))
    height = max(6, 3 + heatmap.shape[0] * 0.2)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(clustered_heatmap.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(clustered_heatmap.shape[1]))
    axis.set_xticklabels(clustered_heatmap.columns, rotation=90, fontsize=5)
    axis.set_yticks(range(clustered_heatmap.shape[0]))
    axis.set_yticklabels(clustered_heatmap.index, fontsize=6)
    axis.set_xlabel("Immune cell type")
    axis.set_ylabel("Gene")
    axis.set_title(
        f"Heatmap for {heatmap.shape[0]} genes\n"
        "hierarchically clustered genes and cell types; "
        "higher ranks are more expression-specific"
    )
    figure.colorbar(image, ax=axis, label="Expression-specificity rank")
    figure.tight_layout()

    output = BytesIO()
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return ""
    if 0 < value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3g}"


st.set_page_config(
    page_title="Immune-cell polygenic expression specificity tester",
    layout="wide",
)

st.title("Immune-cell polygenic expression specificity tester")

with st.sidebar:
    st.markdown(
        "**Test whether a gene set is preferentially expressed across PBMC "
        "cell types in the Immunobiology of Aging Cohort PBMC Profiling "
        "dataset.**"
    )

    gene_text = st.text_area(
        "Input your gene list:",
        value=DEFAULT_GENES,
        height=220,
    )
    background_text = st.text_area(
        "Background gene list (optional):",
        value="",
        height=90,
    )
    species = st.selectbox(
        "Species of input genes:",
        options=["Human", "Mouse", "Rhesus Macaque"],
    )
    st.divider()
    st.markdown("**Source data:**")
    st.markdown(
        "[Multi-omic profiling reveals age-related immune dynamics in healthy "
        "adults](https://www.nature.com/articles/s41586-025-09686-5)  \n"
        "Qiuyu Gong et al."
    )
    st.markdown(
        "[CZ CELLxGENE source collection]"
        "(https://cellxgene.cziscience.com/collections/"
        "60a2676d-9f37-46cc-9b02-c7370a53be9c) from the "
        "[Allen Institute for Immunology]"
        "(https://alleninstitute.org/immunology)."
    )
    st.markdown("**Example gene set:**")
    st.markdown(
        "Default genes are from the [Bellenguez et al. Alzheimer's disease GWAS]"
        "(https://www.nature.com/articles/s41588-022-01024-z) (Table S5)."
    )


saved = None
try:
    parsed_targets = parse_gene_list(gene_text)
    parsed_background = parse_gene_list(background_text)
    if not parsed_targets:
        raise ValueError("Enter at least one target gene.")

    ortholog_map = None if species == "Human" else get_ortholog_map()
    conversion_unmapped = (
        []
        if species == "Human"
        else [
            gene
            for gene in parsed_targets
            if not ortholog_map.get(species, {}).get(gene)
        ]
    )
    target_genes = convert_to_human(parsed_targets, species, ortholog_map)
    background_genes = (
        convert_to_human(parsed_background, species, ortholog_map)
        if parsed_background
        else None
    )
    if not target_genes:
        raise ValueError("None of the input genes could be converted to human symbols.")

    matrix = get_rank_matrix()
    result = run_auroc_analysis(
        matrix=matrix,
        target_genes=target_genes,
        background_genes=background_genes,
    )
    saved = {
        "table": result.table,
        "heatmap": result.heatmap,
        "matched": result.matched_gene_count,
        "input": result.input_gene_count,
        "background": result.background_gene_count,
        "conversion_unmapped": conversion_unmapped,
        "matrix_unmatched": list(result.unmatched_genes),
    }
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))


if saved is not None:
    results_table = saved["table"]

    summary_columns = st.columns([4, 1])
    with summary_columns[0]:
        st.write(f"Genes found in data: {saved['matched']} of {saved['input']}")
    unmapped_count = len(saved["conversion_unmapped"]) + len(
        saved["matrix_unmatched"]
    )
    with summary_columns[1]:
        with st.popover(f"Unmapped genes ({unmapped_count})"):
            if saved["conversion_unmapped"]:
                st.markdown("**Input genes without a human ortholog:**")
                st.code("\n".join(saved["conversion_unmapped"]), language=None)
            if saved["matrix_unmatched"]:
                label = (
                    "Converted human symbols not found in the selected data/background:"
                    if species != "Human"
                    else "Input genes not found in the selected data/background:"
                )
                st.markdown(f"**{label}**")
                st.code("\n".join(saved["matrix_unmatched"]), language=None)
            if unmapped_count == 0:
                st.success("All input genes were mapped and found in the selected data.")
    st.write(f"Background genes: {saved['background']}")

    numeric_columns = ["AUROC", "pValue", "FDR"]
    display_table = (
        results_table.style.format(
            {
                "AUROC": lambda value: f"{value:.3g}",
                "pValue": format_p_value,
                "FDR": format_p_value,
            }
        )
        .set_properties(subset=numeric_columns, **{"text-align": "right"})
    )
    st.html(
        """
        <style>
        [data-testid="stTable"] th.col_heading.level0:not(.col0) {
            text-align: right !important;
        }
        </style>
        """
    )
    st.table(
        display_table,
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "FDR values use the Benjamini-Hochberg correction applied to "
        "full-precision p-values."
    )

    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            "Download results as .csv",
            data=results_table.to_csv(index=False).encode("utf-8"),
            file_name="immune_cell_polygenic_results.csv",
            mime="text/csv",
            on_click="ignore",
            width="stretch",
        )
    with download_columns[1]:
        st.download_button(
            "Download heatmap as .pdf",
            data=create_heatmap_pdf(saved["heatmap"]),
            file_name="immune_cell_polygenic_heatmap.pdf",
            mime="application/pdf",
            on_click="ignore",
            width="stretch",
        )
