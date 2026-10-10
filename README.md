# Data

The trajectory files are not stored in this repository. Download them from Zenodo:

Ali, Y., K. R. Rao, A. Bhaskar, and N. Chatterjee. 2025. Microscopic Vehicle Trajectory Datasets from
UAV-Collected Video for Heterogeneous, Area-Based Urban Traffic. https://doi.org/10.5281/zenodo.17745347

Place the six CSV files (DEL_2, DEL_3, DEL_4, DEL_5, NDA_1, NDA_2) directly in your base folder, the folder
given by `EDA_BASE` (or `BASE` in the notebooks). Each file name must contain the site code, for example
`DEL_2.csv` or `MVT_DEL 3.csv`; the scripts match `DEL` or `NDA` followed by the site number.

After a run, the base folder contains:

    base_folder/
    ├── *.csv                      released files (input)
    ├── EDA_results/               step 1: audit tables, figures, standardised parquet files
    ├── cleaned/                   step 2: cleaned parquet files, cleaning log and report
    └── paper_packs/TRR_combined/  step 3: manuscript and supplement exhibits
