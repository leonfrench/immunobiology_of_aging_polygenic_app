# Immune-cell polygenic expression specificity app

This Streamlit app tests whether an input gene set is preferentially expressed
across immune cell types from healthy adults. It uses the local
`data/cell_type_average_expression.csv.gz` rank matrix and includes
mouse/rhesus-to-human ortholog mappings for the optional species conversion.

The expression data are associated with [*Multi-omic profiling reveals
age-related immune dynamics in healthy adults* by Qiuyu Gong et
al.](https://www.nature.com/articles/s41586-025-09686-5) and the corresponding
[CZ CELLxGENE source collection](https://cellxgene.cziscience.com/collections/60a2676d-9f37-46cc-9b02-c7370a53be9c)
from the [Allen Institute for Immunology](https://alleninstitute.org/immunology).

## Run locally

```bash
cd immunobiology_of_aging_polygenic_app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app supports gene parsing, optional ortholog conversion, optional
custom-background reranking, analytic AUROC and p-values, Benjamini-Hochberg
FDR adjustment of full-precision p-values, CSV download, and PDF heatmap
download with complete-linkage Euclidean clustering of genes and immune cell
types.

The default analysis runs immediately. Editing an input widget automatically
reruns the analysis, and the unmapped-gene popover reports failed ortholog
conversions and genes absent from the selected data or custom background.
