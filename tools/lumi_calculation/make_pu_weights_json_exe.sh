#!/bin/bash

python3 make_pu_weights_json.py \
    --data-nominal MyDataPileupHistogram_NominalXSEC.root \
    --data-up MyDataPileupHistogram_UpXSEC.root \
    --data-down MyDataPileupHistogram_DownXSEC.root \
    --data-hist pileup \
    --mc ${CMSSW_BASE}/src/Reco2Pico/PicoProducer/tools/pileup/pileup_Run3Winter25.root \
    --mc-hist pileup \
    --name Collisions25_goldenJSON \
    -o ./pileupWeightJSON/pileupWeight_Run3Winter25_2025G.json \
    --make-plots \
    --plot-dir plots
