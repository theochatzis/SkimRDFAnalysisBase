#!/bin/bash

set -e

export CORR_BASE=$(
    python3 -c "import correctionlib; print(correctionlib.__path__[0])"
)

g++ \
    -shared \
    -fPIC \
    -o libAnalysisCommon.so \
    src/Corrections.cc \
    src/JECUtils.cc \
    $(root-config --cflags --libs) \
    -I${CORR_BASE}/include \
    -Iinterface \
    -L${CORR_BASE}/lib \
    -lcorrectionlib \
    -Wl,-rpath,${CORR_BASE}/lib

# Keep compatibility with code that still expects the old name.
ln -sf libAnalysisCommon.so libJECUtils.so

echo "Common correction library compiled successfully."
