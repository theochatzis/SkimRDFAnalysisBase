#!/bin/bash
DATA_FILES_DIR=/eos/user/t/tchatzis/reco2pico/myPicosDirectory/windowed_balance_reclusterV2/default/DATA/Muon2025G/

datasetJSON=processed_lumis.json
# Make json file from NanoAOD data files
python3 make_lumi_json.py ${DATA_FILES_DIR} \
    -o ${datasetJSON}

# Make pileup histograms
LumiJSON=/eos/user/c/cmsdqm/www/CAF/certification/Collisions25/PileUp/pileup_JSON-2025_Golden.txt

pileupCalc.py -i ${datasetJSON} --inputLumiJSON ${LumiJSON} --calcMode true --minBiasXsec 69200 --maxPileupBin 100 --numPileupBins 100 MyDataPileupHistogram_NominalXSEC.root

pileupCalc.py -i ${datasetJSON} --inputLumiJSON ${LumiJSON} --calcMode true --minBiasXsec 72400 --maxPileupBin 100 --numPileupBins 100 MyDataPileupHistogram_UpXSEC.root

pileupCalc.py -i ${datasetJSON} --inputLumiJSON ${LumiJSON} --calcMode true --minBiasXsec 66000 --maxPileupBin 100 --numPileupBins 100 MyDataPileupHistogram_DownXSEC.root