#!/bin/bash
DATA_FILES_DIR=/eos/user/t/tchatzis/reco2pico/myPicosDirectory/windowedBalance2024/default/DATA/Muon2024I/ 

# Make json file from NanoAOD data files
python3 make_lumi_json.py ${DATA_FILES_DIR} \
    -o processed_lumis.json