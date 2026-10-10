#!/usr/bin/env bash
# Usage: bash run_all.sh /path/to/base_folder   (folder that holds the six released CSV files)
set -euo pipefail
export EDA_BASE="${1:?give the base folder that holds the CSV files}"
export EDA_HEADLESS=1
here="$(cd "$(dirname "$0")" && pwd)"
python "$here/scripts/01_eda.py"
python "$here/scripts/02_clean.py"
python "$here/scripts/03_paper_exhibits.py"
echo "Done. Paper exhibits are in $EDA_BASE/paper_packs/TRR_combined/"
