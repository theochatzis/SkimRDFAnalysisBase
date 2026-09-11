#!/bin/bash
set -euo pipefail

# Optional in case you have e.g. changed Common/src/Corrections.cc compile once.
(
  cd "$(dirname "$0")/../Common"
  source setup_Common_cpp.sh
)


python3 $(dirname "$0")/../run_analysis.py \
  --input-files-dir /eos/user/t/tchatzis/reco2pico/myPicosDirectory/zjet_window_default/default/DATA/Muon2025G/ \
  --output-dir zjet_example_output \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --add-no-selection

python3 $(dirname "$0")/../run_analysis.py \
  --input-files-dir /eos/user/t/tchatzis/reco2pico/myPicosDirectory/zjet_window_default/default/MC/ZTo2Mu/ \
  --output-dir zjet_example_output \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --add-no-selection

python3 plot_zjet_with_tools.py \
    --data zjet_example_output/Muon2025G.root \
    --mc DY=zjet_example_output/ZTo2Mu.root \
    --region zjet \
    --output-dir zjet_plots \
    --efficiency-process DY

# Optional configuration:
#
# python3 run_analysis.py \
#   --input-files-dir /path/to/sample \
#   --output-dir zjet_example_output \
#   --file-pattern "*.root" \
#   --histograms-defs example_analyses/zjet_histograms.yaml \
#   --rdf-definition example_analyses/zjet_rdf_definition.py \
#   --analysis-config example_analyses/zjet_config.yaml \
#   --add-no-selection
