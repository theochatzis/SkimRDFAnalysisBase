#!/usr/bin/env python3
"""Usage: cmsRun tools/gen_xsec/gen_xsec_cfg.py inputFiles=file:/path/sample.root"""

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.parseArguments()

if not options.inputFiles:
    raise RuntimeError("Pass at least one file with inputFiles=file:/path/file.root")

process = cms.Process("GENXSEC")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source(
    "PoolSource", fileNames=cms.untracked.vstring(options.inputFiles)
)
process.load("GeneratorInterface.Core.genXSecAnalyzer_cfi")
process.p = cms.Path(process.genXSecAnalyzer)
