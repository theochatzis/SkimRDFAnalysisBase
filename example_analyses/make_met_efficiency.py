#!/usr/bin/env python3

import argparse
import os
import sys

from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np


# plot_graphs() renders labels with matplotlib, so these use mathtext rather
# than ROOT TLatex markup.
OFFLINE_VARS = {
    "MET_pt": r"Offline $p^{miss}_{T}$ [GeV]",
    "U_pt": r"Offline $p^{miss}_{T}(no-\mu)$ [GeV]",
    "Z_pt": r"$p^{Z}_{T}$ [GeV]",
}

# The DY sample has far too few events per 5 GeV bin above the turn-on for a
# bin-by-bin efficiency to mean anything, which is also what drives individual
# bins negative through their negative generator weights. Merge bins before
# taking the ratio; 1 leaves the binning as booked in zjet_histograms.yaml.
REBIN = {
    "MET_pt": 4,
    "U_pt": 4,
    "Z_pt": 1,
}

TRIGGER_NAME = "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight"

REPO_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if REPO_DIR not in sys.path:
    sys.path.insert(
        0,
        REPO_DIR,
    )


from plotting import (
    Graph1D,
    Hist1D,
    RootFileReader,
    efficiency_graph,
    hist_to_graph,
    plot_graphs,
    plot_hist1d_data_mc_stack,
)

def rebin(hist, factor):
    """
    Merge `factor` adjacent bins of a Hist1D.

    Bin contents add, sumw2 errors add in quadrature, and every `factor`-th
    edge is kept. There is no rebinning helper in the plotting package, and
    the reader hands back numpy-backed Hist1D objects rather than TH1s.
    """
    if factor <= 1:
        return hist

    if len(hist.values) % factor:
        raise ValueError(
            "Cannot rebin '{}' by {}: {} bins is not a multiple of it.".format(
                hist.name,
                factor,
                len(hist.values),
            )
        )

    errors = None

    if hist.errors is not None:
        errors = np.sqrt(
            np.sum(
                hist.errors.reshape(-1, factor) ** 2,
                axis=1,
            )
        )

    return replace(
        hist,
        values=np.sum(
            hist.values.reshape(-1, factor),
            axis=1,
        ),
        errors=errors,
        edges=hist.edges[::factor],
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--mc",
        required=True,
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--lumi",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--com",
        type=float,
        default=13.6,
    )
    
    args = parser.parse_args()
    
    # Output directory
    output_dir = Path(
        args.output_dir
    )

    def directory(*parts):
        path = output_dir.joinpath(*parts)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
        return path
    
    # Read input files and analyse
    with ExitStack() as stack: # All files openned will automatically close in the end
        # Open the readers
        data_reader = stack.enter_context(
            RootFileReader(
                args.data
            )
        )

        mc_reader = stack.enter_context(
            RootFileReader(
                args.mc
            )
        )
        
        readers = [
            ("Data", data_reader),
            ("DY MC", mc_reader)
        ]

        # Loop over variables
        for var, var_label in OFFLINE_VARS.items():
            efficiency_graphs = OrderedDict()
            
            for key, reader in readers:    
                # Read histograms  
                factor = REBIN.get(var, 1)

                hist_num = rebin(
                    reader.get(f"zjet_{TRIGGER_NAME}/{var}"),
                    factor,
                )
                hist_den = rebin(
                    reader.get(f"zjet/{var}"),
                    factor,
                )
                
                efficiency_graphs[key] = efficiency_graph(
                    hist_num,
                    hist_den,
                    name=f"efficiency_{TRIGGER_NAME}_{var}",
                    tolerance=1.0,
                )
                
            # Make plotting
            plot_graphs(
                efficiency_graphs,
                directory() / f"efficiency_{TRIGGER_NAME}_{var}.png",
                labels={
                    key: key
                    for key in efficiency_graphs
                },
                xlabel=var_label,
                ylabel="Efficiency",
                ylim=[0.0,1.2],
                xlim=[0.0,400.0],
                logx=False,
                reference_line=1.0,
                legend_outside=False,
                cms_label="Preliminary",
                lumi=args.lumi,
                com=args.com,
                data=True,
                title=TRIGGER_NAME,
                title_fontsize=14,
            )

if __name__ == "__main__":
    main()