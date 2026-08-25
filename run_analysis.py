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


# ============================================================
# Python analysis-definition loading
# ============================================================

def resolve_file(path, description):
    """
    Resolve a configuration file.

    First try the path exactly as given (relative to the current working
    directory). If it does not exist, also try relative to this script.
    """
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

    required = ("define_columns", "get_regions")

    for function_name in required:
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
    Convert a YAML edge specification into a numpy array.

    Supported forms:

      edges: [20, 30, 50, 100]

    or

      edges: ["bins", 50, 0, 500]

    In the second form, 50 means 50 bins, therefore 51 edges are made.
    """
    if len(spec) > 0 and spec[0] == "bins":
        nbins = int(spec[1])
        xmin = float(spec[2])
        xmax = float(spec[3])

        return np.linspace(
            xmin,
            xmax,
            nbins + 1,
            dtype=np.float64
        )

    return np.asarray(
        spec,
        dtype=np.float64
    )


def book_histograms(dataframe, config):
    """
    Book all histograms defined in the YAML configuration.

    Optional YAML field:

      weight: "weight_column"

    is supported for TH1D, TProfile, TH2D, TProfile2D and TH3D.
    """
    pointers = []

    for hist_name, hist_info in config.items():
        title = hist_info["title"]
        hist_type = hist_info.get("type", "TH1D")
        weight = hist_info.get("weight")

        # ----------------------------------------------------
        # TH1D
        # ----------------------------------------------------
        if hist_type == "TH1D":
            variable = hist_info["variable"]

            if "edges" in hist_info:
                edges = make_edges(hist_info["edges"])

                model = (
                    hist_name,
                    title,
                    len(edges) - 1,
                    edges
                )

            else:
                bins = hist_info["bins"]

                model = (
                    hist_name,
                    title,
                    bins[0],
                    bins[1],
                    bins[2]
                )

            if weight:
                pointer = dataframe.Histo1D(
                    model,
                    variable,
                    weight
                )
            else:
                pointer = dataframe.Histo1D(
                    model,
                    variable
                )

            pointers.append(pointer)

        # ----------------------------------------------------
        # TProfile / Profile1D
        # ----------------------------------------------------
        elif hist_type in ("TProfile", "Profile1D"):
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]

            if "edges" in hist_info:
                edges = make_edges(hist_info["edges"])

                model = (
                    hist_name,
                    title,
                    len(edges) - 1,
                    edges
                )

            else:
                bins = hist_info["bins"]

                model = (
                    hist_name,
                    title,
                    bins[0],
                    bins[1],
                    bins[2]
                )

            if weight:
                pointer = dataframe.Profile1D(
                    model,
                    var_x,
                    var_y,
                    weight
                )
            else:
                pointer = dataframe.Profile1D(
                    model,
                    var_x,
                    var_y
                )

            pointers.append(pointer)

        # ----------------------------------------------------
        # TH2D
        # ----------------------------------------------------
        elif hist_type == "TH2D":
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]

            if (
                "edges_x" in hist_info
                and "edges_y" in hist_info
            ):
                edges_x = make_edges(hist_info["edges_x"])
                edges_y = make_edges(hist_info["edges_y"])

                model = (
                    hist_name,
                    title,
                    len(edges_x) - 1,
                    edges_x,
                    len(edges_y) - 1,
                    edges_y
                )

            else:
                bins = hist_info["bins"]

                model = (
                    hist_name,
                    title,
                    bins[0],
                    bins[1],
                    bins[2],
                    bins[3],
                    bins[4],
                    bins[5]
                )

            if weight:
                pointer = dataframe.Histo2D(
                    model,
                    var_x,
                    var_y,
                    weight
                )
            else:
                pointer = dataframe.Histo2D(
                    model,
                    var_x,
                    var_y
                )

            pointers.append(pointer)

        # ----------------------------------------------------
        # TH3D
        # ----------------------------------------------------
        elif hist_type == "TH3D":
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]
            var_z = hist_info["variable_z"]

            if (
                "edges_x" in hist_info
                and "edges_y" in hist_info
                and "edges_z" in hist_info
            ):
                edges_x = make_edges(
                    hist_info["edges_x"]
                )
                edges_y = make_edges(
                    hist_info["edges_y"]
                )
                edges_z = make_edges(
                    hist_info["edges_z"]
                )

                model = (
                    hist_name,
                    title,
                    len(edges_x) - 1,
                    edges_x,
                    len(edges_y) - 1,
                    edges_y,
                    len(edges_z) - 1,
                    edges_z,
                )

            else:
                bins = hist_info["bins"]

                model = (
                    hist_name,
                    title,
                    bins[0],
                    bins[1],
                    bins[2],
                    bins[3],
                    bins[4],
                    bins[5],
                    bins[6],
                    bins[7],
                    bins[8],
                )

            if weight:
                pointer = dataframe.Histo3D(
                    model,
                    var_x,
                    var_y,
                    var_z,
                    weight,
                )
            else:
                pointer = dataframe.Histo3D(
                    model,
                    var_x,
                    var_y,
                    var_z,
                )

            pointers.append(
                pointer
            )

        # ----------------------------------------------------
        # TProfile2D / Profile2D
        # ----------------------------------------------------
        elif hist_type in ("TProfile2D", "Profile2D"):
            var_x = hist_info["variable_x"]
            var_y = hist_info["variable_y"]
            var_z = hist_info["variable_z"]

            if (
                "edges_x" in hist_info
                and "edges_y" in hist_info
            ):
                edges_x = make_edges(hist_info["edges_x"])
                edges_y = make_edges(hist_info["edges_y"])

                model = (
                    hist_name,
                    title,
                    len(edges_x) - 1,
                    edges_x,
                    len(edges_y) - 1,
                    edges_y
                )

            else:
                bins = hist_info["bins"]

                model = (
                    hist_name,
                    title,
                    bins[0],
                    bins[1],
                    bins[2],
                    bins[3],
                    bins[4],
                    bins[5]
                )

            if weight:
                pointer = dataframe.Profile2D(
                    model,
                    var_x,
                    var_y,
                    var_z,
                    weight
                )
            else:
                pointer = dataframe.Profile2D(
                    model,
                    var_x,
                    var_y,
                    var_z
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
    """Compile comma-separated regular expressions."""
    return [
        re.compile(pattern.strip())
        for pattern in raw.split(",")
        if pattern.strip()
    ]


def should_process(
    subdir,
    include_patterns,
    skip_patterns
):
    if (
        include_patterns
        and not any(
            pattern.search(subdir)
            for pattern in include_patterns
        )
    ):
        return False

    if (
        skip_patterns
        and any(
            pattern.search(subdir)
            for pattern in skip_patterns
        )
    ):
        return False

    return True


def get_subdirs_at_depth(base_dir, target_depth):
    """
    Return relative subdirectory paths exactly target_depth levels
    below base_dir.

    depth = 0:
        process base_dir itself

    depth = 1:
        process direct children

    depth = 2:
        process grandchildren
    """
    if target_depth == 0:
        return [""]

    base_dir = os.path.abspath(base_dir)
    base_depth = base_dir.rstrip(os.sep).count(os.sep)

    subdirs = []

    for root, dirs, _ in os.walk(base_dir):
        current_depth = (
            root.count(os.sep)
            - base_depth
        )

        if current_depth + 1 == target_depth:
            for directory in dirs:
                full_path = os.path.join(
                    root,
                    directory
                )

                subdirs.append(
                    os.path.relpath(
                        full_path,
                        base_dir
                    )
                )

        if current_depth >= target_depth:
            dirs[:] = []

    return sorted(subdirs)


def find_input_files(directory, pattern):
    """Find ROOT files recursively below directory matching pattern."""
    input_files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                input_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return sorted(input_files)


# ============================================================
# Region validation
# ============================================================

def validate_regions(regions):
    """
    Enforce one simple analysis interface:

      {
          "region_name": {
              "cuts": [
                  "selection 1",
                  "selection 2",
              ]
          }
      }

    No normalization/conversion layer is used.
    """
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
                f"Region '{region_name}' field 'cuts' "
                f"must be a list or tuple."
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
    """Accept true/false for backwards-compatible CLI flags."""
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
            "Physics definitions and regions are supplied by "
            "--rdf-definition."
        )
    )

    # Input
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
        "--input-files-depth",
        type=int,
        default=0,
        help=(
            "Subdirectory depth to treat as separate samples. "
            "0 means process --input-files-dir itself."
        )
    )

    # Analysis
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
        help=(
            "Optional YAML configuration passed to "
            "rdf_definition.py"
        )
    )

    # Output
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

    # Sample selection
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated regex patterns for samples to skip"
    )

    parser.add_argument(
        "--include-only",
        default="",
        help=(
            "Comma-separated regex patterns; process only "
            "matching samples"
        )
    )

    # Event range
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

    # Histograms
    parser.add_argument(
        "--add-no-selection",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help=(
            "Also book histograms before any region cuts. "
            "Supports either '--add-no-selection' or "
            "'--add-no-selection True'."
        )
    )

    # ROOT
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

def make_output_path(
    args,
    relative_subdir,
    sample_name
):
    if args.output_name:
        return os.path.join(
            args.output_dir,
            args.output_name
        )

    if args.input_files_depth == 0:
        return os.path.join(
            args.output_dir,
            f"{sample_name}.root"
        )

    # Preserve the original convention:
    # use the first directory level as output sample name.
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

    # --------------------------------------------------------
    # ROOT multithreading
    # --------------------------------------------------------
    if args.threads > 0:
        ROOT.ROOT.EnableImplicitMT(
            args.threads
        )
    else:
        ROOT.ROOT.EnableImplicitMT()

    print(
        "Threads enabled:",
        ROOT.ROOT.GetThreadPoolSize()
    )

    # --------------------------------------------------------
    # Load analysis definition
    # --------------------------------------------------------
    analysis = load_analysis_definition(
        args.rdf_definition
    )

    # --------------------------------------------------------
    # Load histogram YAML
    # --------------------------------------------------------
    histograms_path = resolve_file(
        args.histograms_defs,
        "Histogram definition"
    )

    with open(
        histograms_path,
        "r",
        encoding="utf-8"
    ) as handle:
        hist_config = (
            yaml.safe_load(handle)
            or {}
        )

    print(
        f"Loaded histogram definitions: "
        f"{histograms_path}"
    )

    # --------------------------------------------------------
    # Optional analysis YAML
    # --------------------------------------------------------
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
            analysis_config = (
                yaml.safe_load(handle)
                or {}
            )

        print(
            f"Loaded analysis configuration: "
            f"{analysis_config_path}"
        )

    # --------------------------------------------------------
    # Optional one-time analysis setup
    #
    # Use this in rdf_definition.py to:
    #   * ROOT.gInterpreter.Declare(...)
    #   * ROOT.gSystem.Load(...)
    #   * initialize correctionlib/JEC helpers
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Find samples
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Sample loop
    # --------------------------------------------------------
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
                f"no files matching "
                f"'{args.file_pattern}'."
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

        # ----------------------------------------------------
        # Base dataframe
        # ----------------------------------------------------
        df = ROOT.RDataFrame(
            args.tree_name,
            input_files
        )

        # Correct Range semantics:
        # Range(begin, end), with end = begin + number to process.
        if args.max_events > 0:
            begin = args.skip_first_nevents
            end = begin + args.max_events

            df = df.Range(
                begin,
                end
            )

        elif args.skip_first_nevents > 0:
            df = df.Range(
                args.skip_first_nevents
            )

        # ----------------------------------------------------
        # Analysis-specific Define/Redefine calls
        # ----------------------------------------------------
        df = analysis.define_columns(
            df,
            sample=sample,
            args=args,
            config=analysis_config
        )

        # ----------------------------------------------------
        # Analysis-specific regions
        # ----------------------------------------------------
        regions = analysis.get_regions(
            sample=sample,
            args=args,
            config=analysis_config
        )

        validate_regions(regions)

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------
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

        # Keep all lazy actions alive until execution.
        histogram_actions = []
        reports = {}

        # ----------------------------------------------------
        # Optional unfiltered histograms
        # ----------------------------------------------------
        if args.add_no_selection:
            print(
                "\nBooking histograms without "
                "region selection"
            )

            pointers = book_histograms(
                df,
                hist_config
            )

            histogram_actions.extend(
                ("", pointer)
                for pointer in pointers
            )

        # ----------------------------------------------------
        # Region loop
        # ----------------------------------------------------
        for region_name, region_info in regions.items():
            print(
                f"\nBooking region: {region_name}"
            )

            region_df = df

            cuts = region_info.get(
                "cuts",
                []
            )

            for cut_index, selection in enumerate(
                cuts
            ):
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

            if not output.GetDirectory(
                region_name
            ):
                output.mkdir(
                    region_name
                )

            pointers = book_histograms(
                region_df,
                hist_config
            )

            histogram_actions.extend(
                (region_name, pointer)
                for pointer in pointers
            )

        # ----------------------------------------------------
        # Execute and write
        #
        # All histogram actions are already booked on the graph.
        # The first action triggers execution; RDF evaluates the
        # booked lazy graph together.
        # ----------------------------------------------------
        print(
            "\nExecuting RDataFrame graph..."
        )

        for target_dir, histogram in histogram_actions:
            if target_dir:
                output.cd(
                    target_dir
                )
            else:
                output.cd()

            histogram.Write()

        # ----------------------------------------------------
        # Cut reports
        # ----------------------------------------------------
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
