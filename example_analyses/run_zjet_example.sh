#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1 # Skip all user-installed Python packages so build and runtime use CMSSW's compatible correctionlib.

# Optional in case you have e.g. changed Common/src/Corrections.cc compile once.
(
  cd "$(dirname "$0")/../Common"
  source setup_Common_cpp.sh
)

TAG="_puWeight69p2"

ROOT_OUTPUT=zjet_output${TAG}
PLOTS_OUTPUT=zjet_main_plots${TAG}
WEIGHTS_OUTPUT=zjet_weights_plots${TAG}
TRIGGER_OUTPUT=zjet_trigger_plots${TAG}

python3 $(dirname "$0")/../run_analysis.py \
  --input-files-dir /eos/user/t/tchatzis/reco2pico/myPicosDirectory/windowed_balance_reclusterV2/default/DATA/Muon2025G/ \
  --output-dir ${ROOT_OUTPUT} \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --weights-defs example_analyses/weights_pu_example.yaml \
  --histogram-weight-output all \
  --add-no-selection


python3 $(dirname "$0")/../run_analysis.py \
  --input-files-dir /eos/user/t/tchatzis/reco2pico/myPicosDirectory/Run3Winter25ZTo2Mu/default/MC/ZTo2Mu/ \
  --output-dir ${ROOT_OUTPUT} \
  --file-pattern "*.root" \
  --histograms-defs example_analyses/zjet_histograms.yaml \
  --rdf-definition example_analyses/zjet_rdf_definition.py \
  --weights-defs example_analyses/weights_pu_example.yaml \
  --histogram-weight-output all \
  --add-no-selection



# --histogram-weight-output all needs --weights-defs to be useful: without a
# weights file there are no scale factors to vary, and the runner writes the
# nominal histograms only.

python3 plot_zjet_with_tools.py \
    --data ${ROOT_OUTPUT}/Muon2025G.root \
    --mc DY=${ROOT_OUTPUT}/ZTo2Mu.root \
    --region zjet \
    --output-dir ${PLOTS_OUTPUT} \
    --efficiency-process DY \
    --mc-scale DY=5798247 # it is (xsec in pb)*5226. (lumi in pb^-1 from brilcalc) * 0.5 becase we run on Muon0 only (half lumi)
#    --normalize-mc-to-data

# Plotting variations
# python3 plot_zjet_variations.py \
#   --input ${ROOT_OUTPUT}/ZTo2Mu.root \
#   --region zjet \
#   --output-dir ${WEIGHTS_OUTPUT}

# Efficiencies plots
./make_met_efficiency.py \
  --data ${ROOT_OUTPUT}/Muon2025G.root \
  --mc ${ROOT_OUTPUT}/ZTo2Mu.root \
  --output-dir ${TRIGGER_OUTPUT}
