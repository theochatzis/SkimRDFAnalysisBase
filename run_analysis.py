#!/usr/bin/env python3

import argparse
import fnmatch
import importlib.util
import os
import re
import time

import numpy as np
import ROOT
import yaml
from tqdm import tqdm

from utils.event_weights import (
    load_weights_config,
    setup_weight_corrections,
    annotate_sample_type,
    apply_event_weights,
    book_weight_histograms,
)


# ============================================================
# Python analysis-definition loading
# ============================================================

def resolve_file(path, description):
    """Resolve a file from cwd first, then relative to this script."""
    path = os.path.expanduser(path)

    if os.path.isfile(path):
        return os.path.abspath(path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_relative = os.path.join(script_dir, path)

    if os.path.isfile(script_relative):
        return os.path.abspath(script_relative)

    raise FileNotFoundError(
        f"{description} not found: {path}\n"
        f"Also tried: {script_relative}"
    )


def load_analysis_definition(path):
    """Import an rdf_definition.py-style analysis module from a file path."""
    path = resolve_file(path, "Analysis definition")

    spec = importlib.util.spec_from_file_location(
        "rdf_analysis_definition",
        path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import analysis definition: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for function_name in ("define_columns", "get_regions"):
        function = getattr(module, function_name, None)

        if not callable(function):
            raise RuntimeError(
                f"Analysis definition '{path}' must define "
                f"a callable {function_name}()."
            )

    return module


# ============================================================
# Histogram helpers
# ============================================================

def make_edges(spec):
    """
    Supported:
      edges: [20, 30, 50, 100]
      edges: ["bins", 50, 0, 500]
    """
    if len(spec) > 0 and spec[0] == "bins":
        return np.linspace(
            float(spec[2]),
            float(spec[3]),
            int(spec[1]) + 1,
            dtype=np.float64
        )

    return np.asarray(spec, dtype=np.float64)


def _histogram_weight(hist_info, default_weight):
    """
    If a histogram explicitly contains `weight`, respect it.
    Therefore:
      no `weight:` field  -> use the global event weight
      weight: null        -> explicitly unweighted
      weight: someColumn  -> use that column
    """
    if "weight" in hist_info:
        return hist_info["weight"]

    return default_weight


def book_histograms(dataframe, config, default_weight=None):
    """
    Book all YAML-defined histograms.

    `default_weight` is normally the combined eventWeight column from
    weights.yaml. A histogram-level `weight:` entry overrides it.
    """
    pointers = []

    for hist_name, hist_info in config.items():
        title = hist_info["title"]
        hist_type = hist_info.get("type", "TH1D")
        weight = _histogram_weight(hist_info, default_weight)

        if hist_type == "TH1D":
            variable = hist_info["variable"]

            if "edges" in hist_info:
                edges = make_edges(hist_info["edges"])
                model = (hist_name, title, len(edges) - 1, edges)
            else:
                bins = hist_info["bins"]
                model = (hist_name, title, bins[0], bins[1], bins[2])

            pointer = (
                dataframe.Histo1D(model, variable, weight)
                if weight
                else dataframe.Histo1D(model, variable)
            )
            pointers.append(pointer)

        elif hist_type in ("TProfile", "Profile1D"):
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]

            if "edges" in hist_info:
                edges = make_edges(hist_info["edges"])
                model = (hist_name, title, len(edges) - 1, edges)
            else:
                bins = hist_info["bins"]
                model = (hist_name, title, bins[0], bins[1], bins[2])

            pointer = (
                dataframe.Profile1D(model, var_x, var_y, weight)
                if weight
                else dataframe.Profile1D(model, var_x, var_y)
            )
            pointers.append(pointer)

        elif hist_type == "TH2D":
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]

            if "edges_x" in hist_info and "edges_y" in hist_info:
                edges_x = make_edges(hist_info["edges_x"])
                edges_y = make_edges(hist_info["edges_y"])
                model = (
                    hist_name, title,
                    len(edges_x) - 1, edges_x,
                    len(edges_y) - 1, edges_y
                )
            else:
                bins = hist_info["bins"]
                model = (
                    hist_name, title,
                    bins[0], bins[1], bins[2],
                    bins[3], bins[4], bins[5]
                )

            pointer = (
                dataframe.Histo2D(model, var_x, var_y, weight)
                if weight
                else dataframe.Histo2D(model, var_x, var_y)
            )
            pointers.append(pointer)

        elif hist_type == "TH3D":
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]
            var_z = hist_info["variable_z"]

            if (
                "edges_x" in hist_info
                and "edges_y" in hist_info
                and "edges_z" in hist_info
            ):
                edges_x = make_edges(hist_info["edges_x"])
                edges_y = make_edges(hist_info["edges_y"])
                edges_z = make_edges(hist_info["edges_z"])

                model = (
                    hist_name, title,
                    len(edges_x) - 1, edges_x,
                    len(edges_y) - 1, edges_y,
                    len(edges_z) - 1, edges_z
                )
            else:
                bins = hist_info["bins"]
                model = (
                    hist_name, title,
                    bins[0], bins[1], bins[2],
                    bins[3], bins[4], bins[5],
                    bins[6], bins[7], bins[8]
                )

            pointer = (
                dataframe.Histo3D(model, var_x, var_y, var_z, weight)
                if weight
                else dataframe.Histo3D(model, var_x, var_y, var_z)
            )
            pointers.append(pointer)

        elif hist_type in ("TProfile2D", "Profile2D"):
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]
            var_z = hist_info["variable_z"]

            if "edges_x" in hist_info and "edges_y" in hist_info:
                edges_x = make_edges(hist_info["edges_x"])
                edges_y = make_edges(hist_info["edges_y"])
                model = (
                    hist_name, title,
                    len(edges_x) - 1, edges_x,
                    len(edges_y) - 1, edges_y
                )
            else:
                bins = hist_info["bins"]
                model = (
                    hist_name, title,
                    bins[0], bins[1], bins[2],
                    bins[3], bins[4], bins[5]
                )

            pointer = (
                dataframe.Profile2D(model, var_x, var_y, var_z, weight)
                if weight
                else dataframe.Profile2D(model, var_x, var_y, var_z)
            )
            pointers.append(pointer)

        else:
            print(
                f"WARNING: Unknown histogram type "
                f"'{hist_type}' for '{hist_name}'. Skipping."
            )

    return pointers


# ============================================================
# Input sample helpers
# ============================================================

def compile_patterns(raw):
    return [
        re.compile(pattern.strip())
        for pattern in raw.split(",")
        if pattern.strip()
    ]


def should_process(subdir, include_patterns, skip_patterns):
    if (
        include_patterns
        and not any(pattern.search(subdir) for pattern in include_patterns)
    ):
        return False

    if (
        skip_patterns
        and any(pattern.search(subdir) for pattern in skip_patterns)
    ):
        return False

    return True


def get_subdirs_at_depth(base_dir, target_depth):
    if target_depth == 0:
        return [""]

    base_dir = os.path.abspath(base_dir)
    base_depth = base_dir.rstrip(os.sep).count(os.sep)
    subdirs = []

    for root, dirs, _ in os.walk(base_dir):
        current_depth = root.count(os.sep) - base_depth

        if current_depth + 1 == target_depth:
            for directory in dirs:
                full_path = os.path.join(root, directory)
                subdirs.append(os.path.relpath(full_path, base_dir))

        if current_depth >= target_depth:
            dirs[:] = []

    return sorted(subdirs)


def find_input_files(directory, pattern):
    input_files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                input_files.append(os.path.join(root, filename))

    return sorted(input_files)


def _read_runs_sumw(input_file, tree_name, branch_name):
    """Return a Runs-tree weight sum, or None if the source is unavailable."""
    root_file = ROOT.TFile.Open(input_file)

    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open input ROOT file: {input_file}")

    try:
        runs = root_file.Get(tree_name)

        if not runs or not runs.InheritsFrom("TTree"):
            return None

        if not runs.GetBranch(branch_name):
            return None

        return sum(float(getattr(entry, branch_name)) for entry in runs)
    finally:
        root_file.Close()


def _read_histogram_sumw(input_file, histogram_name):
    """Return a histogram weight sum, or None if the source is unavailable."""
    root_file = ROOT.TFile.Open(input_file)

    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open input ROOT file: {input_file}")

    try:
        histogram = root_file.Get(histogram_name)

        if not histogram or not histogram.InheritsFrom("TH1"):
            return None

        return float(histogram.GetSumOfWeights())
    finally:
        root_file.Close()


def get_mc_normalization(input_files, runs_tree, runs_branch, histogram_name):
    """Return 1 / sum(genEventSumw), preferring Runs over a histogram."""
    runs_values = [
        _read_runs_sumw(input_file, runs_tree, runs_branch)
        for input_file in input_files
    ]

    if all(value is not None for value in runs_values):
        source = f"{runs_tree}.{runs_branch}"
        total_sumw = sum(runs_values)
    else:
        histogram_values = [
            _read_histogram_sumw(input_file, histogram_name)
            for input_file in input_files
        ]

        if not all(value is not None for value in histogram_values):
            raise RuntimeError(
                "MC normalization requires either "
                f"'{runs_tree}.{runs_branch}' in every input file or "
                f"histogram '{histogram_name}' in every input file."
            )

        source = f"histogram '{histogram_name}'"
        total_sumw = sum(histogram_values)

    if total_sumw == 0.0:
        raise RuntimeError(
            f"MC normalization from {source} is zero; cannot scale histograms."
        )

    return 1.0 / total_sumw, source, total_sumw


def is_profile(histogram):
    return (
        histogram.InheritsFrom("TProfile")
        or histogram.InheritsFrom("TProfile2D")
    )


# ============================================================
# Region validation
# ============================================================

def validate_regions(regions):
    if not isinstance(regions, dict):
        raise TypeError(
            "get_regions() must return a dictionary."
        )

    for region_name, region_info in regions.items():
        if not isinstance(region_info, dict):
            raise TypeError(
                f"Region '{region_name}' must be a dictionary."
            )

        cuts = region_info.get("cuts", [])

        if not isinstance(cuts, (list, tuple)):
            raise TypeError(
                f"Region '{region_name}' field 'cuts' must be a list or tuple."
            )

        for selection in cuts:
            if not isinstance(selection, str):
                raise TypeError(
                    f"Region '{region_name}' contains a "
                    f"non-string selection: {selection!r}"
                )


# ============================================================
# CLI
# ============================================================

def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in ("true", "1", "yes", "y"):
        return True

    if value in ("false", "0", "no", "n"):
        return False

    raise argparse.ArgumentTypeError(
        f"Expected true/false, got '{value}'."
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generic ROOT RDataFrame analysis runner. "
            "Physics definitions and regions are supplied by --rdf-definition."
        )
    )

    parser.add_argument(
        "--input-files-dir",
        required=True,
        help="Directory containing input ROOT files/subdirectories"
    )
    parser.add_argument(
        "--file-pattern",
        default="*Skim*.root",
        help="Input filename pattern, e.g. '*Skim*.root'"
    )
    parser.add_argument(
        "--tree-name",
        default="Events",
        help="Input TTree name"
    )
    parser.add_argument(
        "--runs-tree-name",
        default="Runs",
        help="Runs TTree used for MC normalization"
    )
    parser.add_argument(
        "--runs-sumw-branch",
        default="genEventSumw",
        help="Runs TTree branch containing the generated-event sum of weights"
    )
    parser.add_argument(
        "--gen-event-weight-histogram",
        default="GenEventWeight",
        help="Fallback histogram used for MC normalization"
    )
    parser.add_argument(
        "--input-files-depth",
        type=int,
        default=0,
        help=(
            "Subdirectory depth to treat as separate samples. "
            "0 means process --input-files-dir itself."
        )
    )

    parser.add_argument(
        "--rdf-definition",
        required=True,
        help=(
            "Python file providing setup(), define_columns() "
            "and get_regions()"
        )
    )
    parser.add_argument(
        "--histograms-defs",
        required=True,
        help="YAML file defining histograms"
    )
    parser.add_argument(
        "--analysis-config",
        default="",
        help="Optional YAML configuration passed to rdf_definition.py"
    )
    parser.add_argument(
        "--weights-defs",
        "--weights",
        dest="weights_defs",
        default="",
        help="Optional YAML file defining event weights and their variations"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output ROOT files"
    )
    parser.add_argument(
        "--output-name",
        default="",
        help=(
            "Explicit output ROOT filename. Best used when "
            "processing one sample."
        )
    )

    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated regex patterns for samples to skip"
    )
    parser.add_argument(
        "--include-only",
        default="",
        help="Comma-separated regex patterns; process only matching samples"
    )

    parser.add_argument(
        "--skip-first-nevents",
        type=int,
        default=0,
        help="Skip the first N events"
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Maximum number of events to process; -1 means all"
    )

    parser.add_argument(
        "--add-no-selection",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help=(
            "Also book histograms before any region cuts. "
            "Supports '--add-no-selection' or '--add-no-selection True'."
        )
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "Number of ROOT implicit-MT threads. "
            "0 lets ROOT choose."
        )
    )

    return parser


# ============================================================
# Output naming
# ============================================================

def make_output_path(args, relative_subdir, sample_name):
    if args.output_name:
        return os.path.join(args.output_dir, args.output_name)

    if args.input_files_depth == 0:
        return os.path.join(
            args.output_dir,
            f"{sample_name}.root"
        )

    first_level = relative_subdir.split(os.sep)[0]

    return os.path.join(
        args.output_dir,
        f"{first_level}.root"
    )


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()

    parser = build_parser()
    args = parser.parse_args()

    if args.threads > 0:
        ROOT.ROOT.EnableImplicitMT(args.threads)
    else:
        ROOT.ROOT.EnableImplicitMT()

    print(
        "Threads enabled:",
        ROOT.ROOT.GetThreadPoolSize()
    )

    analysis = load_analysis_definition(
        args.rdf_definition
    )

    histograms_path = resolve_file(
        args.histograms_defs,
        "Histogram definition"
    )

    with open(
        histograms_path,
        "r",
        encoding="utf-8"
    ) as handle:
        hist_config = yaml.safe_load(handle) or {}

    print(
        f"Loaded histogram definitions: "
        f"{histograms_path}"
    )

    analysis_config = {}

    if args.analysis_config:
        analysis_config_path = resolve_file(
            args.analysis_config,
            "Analysis configuration"
        )

        with open(
            analysis_config_path,
            "r",
            encoding="utf-8"
        ) as handle:
            analysis_config = yaml.safe_load(handle) or {}

        print(
            f"Loaded analysis configuration: "
            f"{analysis_config_path}"
        )

    weights_config = {}

    if args.weights_defs:
        weights_config, weights_path = load_weights_config(
            args.weights_defs,
            resolve_file
        )

        print(
            f"Loaded weight definitions: "
            f"{weights_path}"
        )

    setup = getattr(
        analysis,
        "setup",
        None
    )

    if callable(setup):
        setup(
            args=args,
            config=analysis_config
        )

    if weights_config:
        repo_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        setup_weight_corrections(
            weights_config,
            repo_dir
        )

    include_patterns = compile_patterns(
        args.include_only
    )
    skip_patterns = compile_patterns(
        args.skip
    )

    relative_subdirs = [
        subdir
        for subdir in get_subdirs_at_depth(
            args.input_files_dir,
            args.input_files_depth
        )
        if should_process(
            subdir,
            include_patterns,
            skip_patterns
        )
    ]

    print(
        "Samples to process:",
        relative_subdirs
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    for relative_subdir in tqdm(
        relative_subdirs,
        desc="Processing samples"
    ):
        full_subdir_path = os.path.join(
            args.input_files_dir,
            relative_subdir
        )

        if relative_subdir == "":
            sample_name = os.path.basename(
                os.path.abspath(
                    args.input_files_dir
                )
            )
        else:
            sample_name = relative_subdir

        print(
            "\n"
            "============================================================"
        )
        print(
            f"Processing sample: {sample_name}"
        )
        print(
            f"Input directory: {full_subdir_path}"
        )

        input_files = find_input_files(
            full_subdir_path,
            args.file_pattern
        )

        if not input_files:
            print(
                f"Skipping {sample_name}: "
                f"no files matching '{args.file_pattern}'."
            )
            continue

        print(
            f"Found {len(input_files)} input file(s)."
        )

        sample = {
            "name": sample_name,
            "relative_path": relative_subdir,
            "input_dir": full_subdir_path,
            "input_files": input_files,
        }

        df = ROOT.RDataFrame(
            args.tree_name,
            input_files
        )

        annotate_sample_type(
            df,
            sample
        )

        if sample["is_mc"]:
            normalization, source, total_sumw = get_mc_normalization(
                input_files,
                args.runs_tree_name,
                args.runs_sumw_branch,
                args.gen_event_weight_histogram,
            )
            sample["mc_normalization"] = normalization

            print(
                f"MC normalization: 1 / {total_sumw:g} "
                f"(from {source})"
            )

        print(
            "Sample type:",
            "MC" if sample["is_mc"] else "Data"
        )

        if args.max_events > 0:
            begin = args.skip_first_nevents
            end = begin + args.max_events
            df = df.Range(begin, end)

        elif args.skip_first_nevents > 0:
            df = df.Range(
                args.skip_first_nevents
            )

        df = analysis.define_columns(
            df,
            sample=sample,
            args=args,
            config=analysis_config
        )

        weight_state = {
            "enabled": False,
            "event_weight": None,
            "active": [],
            "variations": {},
        }

        if weights_config:
            df, weight_state = apply_event_weights(
                df,
                weights_config,
                sample
            )

        define_weighted_columns = getattr(analysis, "define_weighted_columns", None)
        if callable(define_weighted_columns):
            df = define_weighted_columns(df, sample=sample, args=args, config=analysis_config,
                                         event_weight=weight_state["event_weight"])

        default_hist_weight = (
            weight_state["event_weight"]
            if weight_state["enabled"]
            else None
        )

        regions = analysis.get_regions(
            sample=sample,
            args=args,
            config=analysis_config
        )

        validate_regions(
            regions
        )

        output_path = make_output_path(
            args,
            relative_subdir,
            sample_name
        )

        output = ROOT.TFile(
            output_path,
            "RECREATE"
        )

        if not output or output.IsZombie():
            raise RuntimeError(
                f"Could not create output ROOT file: "
                f"{output_path}"
            )

        histogram_actions = []
        reports = {}

        # Weight-monitoring histograms are intentionally unweighted and
        # are booked before region cuts.
        if weight_state["enabled"]:
            weight_actions = book_weight_histograms(
                df,
                weights_config,
                weight_state
            )

            if weight_actions:
                if not output.GetDirectory("weights"):
                    output.mkdir("weights")

                histogram_actions.extend(
                    ("weights", pointer)
                    for pointer in weight_actions
                )

        if args.add_no_selection:
            print(
                "\nBooking histograms without "
                "region selection"
            )

            pointers = book_histograms(
                df,
                hist_config,
                default_weight=default_hist_weight
            )

            histogram_actions.extend(
                ("", pointer)
                for pointer in pointers
            )

        for region_name, region_info in regions.items():
            print(
                f"\nBooking region: {region_name}"
            )

            region_df = df
            cuts = region_info.get(
                "cuts",
                []
            )

            for cut_index, selection in enumerate(cuts):
                cut_name = (
                    f"{region_name}:"
                    f"{cut_index + 1}: "
                    f"{selection}"
                )

                region_df = region_df.Filter(
                    selection,
                    cut_name
                )

            reports[region_name] = (
                region_df.Report()
            )

            if not output.GetDirectory(region_name):
                output.mkdir(region_name)

            pointers = book_histograms(
                region_df,
                hist_config,
                default_weight=default_hist_weight
            )

            histogram_actions.extend(
                (region_name, pointer)
                for pointer in pointers
            )

        print(
            "\nExecuting RDataFrame graph..."
        )

        for target_dir, histogram in histogram_actions:
            if target_dir:
                output.cd(target_dir)
            else:
                output.cd()

            root_histogram = histogram.GetValue()

            if (
                sample["is_mc"]
                and target_dir != "weights"
                and not is_profile(root_histogram)
            ):
                root_histogram.Scale(sample["mc_normalization"])

            root_histogram.Write()

        for region_name, report in reports.items():
            print(
                f"\n--- Cut report: "
                f"{region_name} ---"
            )

            report.Print()

        output.Close()

        print(
            f"\nOutput written: "
            f"{output_path}"
        )

    print(
        f"\nTotal runtime: "
        f"{time.time() - start_time:.2f} s"
    )


if __name__ == "__main__":
    main()
