#!/usr/bin/env python3

import argparse
import fnmatch
import importlib.util
import os
import re
import time
from pathlib import Path

import numpy as np
import ROOT
import yaml
from tqdm import tqdm


def load_python_module(path, module_name="rdf_definition"):
    """Import a python file by path and return the loaded module."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Analysis definition not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import analysis definition: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_analysis_module(module):
    required = ["define_columns", "get_regions"]
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            "rdf definition is missing required callable(s): "
            + ", ".join(missing)
        )


def book_histograms(dataframe, config):
    """Book all histograms from a YAML dictionary on an RDataFrame node."""
    pointers = []

    for hist_name, hist_info in config.items():
        title = hist_info["title"]
        hist_type = hist_info.get("type", "TH1D")

        if hist_type == "TH1D":
            variable = hist_info["variable"]
            if "edges" in hist_info:
                edges = make_edges(hist_info["edges"])
                model = (hist_name, title, len(edges) - 1, edges)
            else:
                bins = hist_info["bins"]
                model = (hist_name, title, bins[0], bins[1], bins[2])
            pointers.append(dataframe.Histo1D(model, variable))

        elif hist_type in ("TProfile", "Profile1D"):
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]
            if "edges" in hist_info:
                edges = make_edges(hist_info["edges"])
                model = (hist_name, title, len(edges) - 1, edges)
            else:
                bins = hist_info["bins"]
                model = (hist_name, title, bins[0], bins[1], bins[2])
            pointers.append(dataframe.Profile1D(model, var_x, var_y))

        elif hist_type == "TH2D":
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]

            if "edges_x" in hist_info and "edges_y" in hist_info:
                edges_x = make_edges(hist_info["edges_x"])
                edges_y = make_edges(hist_info["edges_y"])
                model = (
                    hist_name,
                    title,
                    len(edges_x) - 1,
                    edges_x,
                    len(edges_y) - 1,
                    edges_y,
                )
            else:
                bins = hist_info["bins"]
                model = (
                    hist_name,
                    title,
                    bins[0], bins[1], bins[2],
                    bins[3], bins[4], bins[5],
                )
            pointers.append(dataframe.Histo2D(model, var_x, var_y))

        elif hist_type in ("TProfile2D", "Profile2D"):
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]
            var_z = hist_info["variable_z"]

            if "edges_x" in hist_info and "edges_y" in hist_info:
                edges_x = make_edges(hist_info["edges_x"])
                edges_y = make_edges(hist_info["edges_y"])
                model = (
                    hist_name,
                    title,
                    len(edges_x) - 1,
                    edges_x,
                    len(edges_y) - 1,
                    edges_y,
                )
            else:
                bins = hist_info["bins"]
                model = (
                    hist_name,
                    title,
                    bins[0], bins[1], bins[2],
                    bins[3], bins[4], bins[5],
                )
            pointers.append(dataframe.Profile2D(model, var_x, var_y, var_z))

        else:
            print(f"WARNING: unknown histogram type '{hist_type}', skipping {hist_name}")

    return pointers


def make_edges(spec):
    """
    YAML syntax:
      edges: [20, 30, 50, 100]
    or
      edges: ["bins", 50, 0, 500]

    For the second form the integer is the number of bins, so n+1 edges are made.
    """
    if spec[0] == "bins":
        nbins, xmin, xmax = int(spec[1]), float(spec[2]), float(spec[3])
        return np.linspace(xmin, xmax, nbins + 1, dtype=np.float64)
    return np.asarray(spec, dtype=np.float64)


def compile_patterns(raw):
    return [re.compile(p.strip()) for p in raw.split(",") if p.strip()]


def should_process(subdir, include_patterns, skip_patterns):
    if include_patterns and not any(p.search(subdir) for p in include_patterns):
        return False
    if skip_patterns and any(p.search(subdir) for p in skip_patterns):
        return False
    return True


def get_subdirs_at_depth(base_dir, target_depth):
    if target_depth == 0:
        return [""]

    base_dir = os.path.abspath(base_dir)
    subdirs = []
    base_depth = base_dir.rstrip(os.sep).count(os.sep)

    for root, dirs, _ in os.walk(base_dir):
        current_depth = root.count(os.sep) - base_depth

        if current_depth + 1 == target_depth:
            for directory in dirs:
                subdirs.append(os.path.relpath(os.path.join(root, directory), base_dir))

        if current_depth >= target_depth:
            dirs[:] = []

    return sorted(subdirs)


def find_input_files(directory, pattern):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                files.append(os.path.join(root, filename))
    return sorted(files)


def define_regions(regions):
    """
    Accepted forms:

      {
        "baseline": ["cut1", "cut2"],
        "signal": ["cut1", "cut3"]
      }

    or

      {
        "baseline": {"cuts": [...]},
        "signal": {"cuts": [...]}
      }
    """
    defined_regions = {}

    for name, cfg in regions.items():
        if isinstance(cfg, (list, tuple)):
            defined_regions[name] = {"cuts": list(cfg)}
        elif isinstance(cfg, dict):
            defined_regions[name] = dict(cfg)
            defined_regions[name].setdefault("cuts", [])
        else:
            raise TypeError(f"Region '{name}' must be a list or dictionary")

    return defined_regions


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generic ROOT RDataFrame analysis runner"
    )

    parser.add_argument("--input-files-dir", required=True)
    parser.add_argument("--file-pattern", default="*.root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--tree-name", default="Events")

    parser.add_argument("--histograms-defs", required=True)
    parser.add_argument(
        "--rdf-definition",
        required=True,
        help="Python analysis module defining columns and regions",
    )

    parser.add_argument("--skip", default="")
    parser.add_argument("--include-only", default="")
    parser.add_argument("--skip-first-nevents", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--input-files-depth", type=int, default=0)
    parser.add_argument("--add-no-selection", action="store_true")

    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ROOT implicit-MT threads. 0 lets ROOT choose.",
    )

    # Generic free-form analysis configuration.
    parser.add_argument(
        "--analysis-config",
        default="",
        help="Optional YAML passed to rdf_definition.py",
    )

    return parser


def main():
    t0 = time.time()
    parser = build_parser()
    args = parser.parse_args()

    if args.threads > 0:
        ROOT.ROOT.EnableImplicitMT(args.threads)
    else:
        ROOT.ROOT.EnableImplicitMT()

    print("Threads enabled:", ROOT.ROOT.GetThreadPoolSize())

    analysis = load_python_module(args.rdf_definition)
    validate_analysis_module(analysis)

    with open(args.histograms_defs) as handle:
        hist_config = yaml.safe_load(handle) or {}

    analysis_config = {}
    if args.analysis_config:
        with open(args.analysis_config) as handle:
            analysis_config = yaml.safe_load(handle) or {}

    # Optional one-time hook for C++ libraries, correctionlib, declarations, etc.
    setup = getattr(analysis, "setup", None)
    if callable(setup):
        setup(args=args, config=analysis_config)

    include_patterns = compile_patterns(args.include_only)
    skip_patterns = compile_patterns(args.skip)

    subdirs = [
        d for d in get_subdirs_at_depth(args.input_files_dir, args.input_files_depth)
        if should_process(d, include_patterns, skip_patterns)
    ]

    print("Samples:", subdirs)

    os.makedirs(args.output_dir, exist_ok=True)

    for relative_subdir in tqdm(subdirs, desc="Processing samples"):
        full_subdir_path = os.path.join(args.input_files_dir, relative_subdir)

        sample_name = (
            os.path.basename(os.path.abspath(args.input_files_dir))
            if relative_subdir == ""
            else relative_subdir
        )

        print("\n================================================================")
        print(f"Processing sample: {sample_name}")
        print(f"Input directory: {full_subdir_path}")

        input_files = find_input_files(full_subdir_path, args.file_pattern)

        if not input_files:
            print(f"Skipping {sample_name}: no matching ROOT files")
            continue

        sample = {
            "name": sample_name,
            "relative_path": relative_subdir,
            "input_dir": full_subdir_path,
            "input_files": input_files,
        }

        df = ROOT.RDataFrame(args.tree_name, input_files)

        if args.max_events > 0:
            begin = args.skip_first_nevents
            end = begin + args.max_events
            df = df.Range(begin, end)
        elif args.skip_first_nevents > 0:
            df = df.Range(args.skip_first_nevents)

        # All physics-specific Define/Redefine calls live here.
        df = analysis.define_columns(
            df,
            sample=sample,
            args=args,
            config=analysis_config,
        )

        regions = define_regions(
            analysis.get_regions(
                sample=sample,
                args=args,
                config=analysis_config,
            )
        )

        if args.output_name:
            output_path = os.path.join(args.output_dir, args.output_name)
        elif args.input_files_depth == 0:
            output_path = os.path.join(args.output_dir, f"{sample_name}.root")
        else:
            first_level = relative_subdir.split(os.sep)[0]
            output_path = os.path.join(args.output_dir, f"{first_level}.root")

        output = ROOT.TFile(output_path, "RECREATE")

        all_hist_pointers = []
        reports = {}

        if args.add_no_selection:
            print("\nBooking histograms without region selection")
            baseline = book_histograms(df, hist_config)
            all_hist_pointers.extend([("", ptr) for ptr in baseline])

        for region_name, region_info in regions.items():
            print(f"\nBooking region: {region_name}")
            region_df = df

            for idx, selection in enumerate(region_info.get("cuts", [])):
                cut_name = f"{region_name}:{idx}:{selection}"
                region_df = region_df.Filter(selection, cut_name)

            reports[region_name] = region_df.Report()

            output.mkdir(region_name)
            region_pointers = book_histograms(region_df, hist_config)
            all_hist_pointers.extend(
                [(region_name, ptr) for ptr in region_pointers]
            )

        print("\nExecuting RDataFrame graph...")

        # Lazy actions share the same graph. First GetValue/Write triggers event loop.
        for target_dir, hist in all_hist_pointers:
            output.cd(target_dir) if target_dir else output.cd()
            hist.Write()

        for region_name, report in reports.items():
            print(f"\n--- Cut report: {region_name} ---")
            report.Print()

        output.Close()
        print(f"\nOutput written: {output_path}")

    print(f"\nTotal runtime: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
