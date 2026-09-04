#!/bin/bash

# A Z+Jet analysis example.
# Data
python3 run_analysis.py \
  --input-files-dir /eos/user/t/tchatzis/reco2pico/myPicosDirectory/zjet_window_default/default/DATA/Muon2025G/ \
  --output-dir zjet_example_output \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --add-no-selection

# MC
python3 run_analysis.py \
  --input-files-dir /eos/user/t/tchatzis/reco2pico/myPicosDirectory/zjet_window_default/default/MC/ZTo2Mu/ \
  --output-dir zjet_example_output \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --add-no-selection
