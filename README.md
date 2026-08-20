# Skims Analysis Base
Basic tools to analyze ntuples skims for studies. 

# Setup
Compile the cpp libraries by using:
``` 
bash setup.sh
```

# `run_analysis.py` script
The `run_analysis.py` features YAML-defined region and histogram configurations in order to produce in the end histograms. Each region is a set of cuts.


In addition can so far natively integrates Python's correctionlib via a compiled C++ shared library to apply Jet Energy Corrections (JECs) and Type-1 MET corrections on the fly.


