#!/usr/bin/env python3

"""
Generic ROOT-object plotting CLI for SkimRDFAnalysisBase.

Examples
--------

TH1/TProfile Data vs MC:
  python plot_root_objects.py compare \
      --data data.root --mc mc.root \
      --object zjet/Jet_eta_parallel \
      --output jet_eta.pdf

TGraph/TEfficiency Data vs MC:
  python plot_root_objects.py compare \
      --data data.root --mc mc.root \
      --object efficiency/myEfficiency \
      --output efficiency.pdf

TH2 Data/MC/ratio:
  python plot_root_objects.py compare2d \
      --data data.root --mc mc.root \
      --object response/h2 \
      --output h2_compare.pdf
"""

import argparse

from plotting import (
    Hist1D,
    Hist2D,
    Graph1D,
    read_root_object,
    plot_hist1d_data_mc,
    plot_graph_data_mc,
    plot_hist2d_data_mc,
)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--data", required=True)
    compare.add_argument("--mc", required=True)
    compare.add_argument("--object", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--xlabel", default=None)
    compare.add_argument("--ylabel", default=None)
    compare.add_argument("--normalize", action="store_true")
    compare.add_argument("--normalize-mc-to-data", action="store_true")
    compare.add_argument("--logy", action="store_true")
    compare.add_argument("--cms-label", default="Preliminary")
    compare.add_argument("--lumi", type=float, default=None)
    compare.add_argument("--com", type=float, default=13.6)

    compare2d = sub.add_parser("compare2d")
    compare2d.add_argument("--data", required=True)
    compare2d.add_argument("--mc", required=True)
    compare2d.add_argument("--object", required=True)
    compare2d.add_argument("--output", required=True)
    compare2d.add_argument("--xlabel", default=None)
    compare2d.add_argument("--ylabel", default=None)
    compare2d.add_argument("--cms-label", default="Preliminary")
    compare2d.add_argument("--lumi", type=float, default=None)
    compare2d.add_argument("--com", type=float, default=13.6)

    args = parser.parse_args()

    data = read_root_object(args.data, args.object)
    mc = read_root_object(args.mc, args.object)

    if args.command == "compare":
        if isinstance(data, Hist1D) and isinstance(mc, Hist1D):
            plot_hist1d_data_mc(
                data,
                mc,
                args.output,
                xlabel=args.xlabel,
                ylabel=args.ylabel or "Events",
                normalize=args.normalize,
                normalize_mc_to_data=args.normalize_mc_to_data,
                logy=args.logy,
                cms_label=args.cms_label,
                lumi=args.lumi,
                com=args.com,
            )
            return

        if isinstance(data, Graph1D) and isinstance(mc, Graph1D):
            plot_graph_data_mc(
                data,
                mc,
                args.output,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                cms_label=args.cms_label,
                lumi=args.lumi,
                com=args.com,
            )
            return

        raise TypeError(
            f"Unsupported compare combination: "
            f"{type(data).__name__}, {type(mc).__name__}"
        )

    if args.command == "compare2d":
        if not isinstance(data, Hist2D) or not isinstance(mc, Hist2D):
            raise TypeError("compare2d requires TH2/TProfile2D objects.")

        plot_hist2d_data_mc(
            data,
            mc,
            args.output,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            cms_label=args.cms_label,
            lumi=args.lumi,
            com=args.com,
        )


if __name__ == "__main__":
    main()
