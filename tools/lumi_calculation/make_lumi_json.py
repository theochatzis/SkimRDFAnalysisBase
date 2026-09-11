#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from tqdm import tqdm

# ----------------------------------------------------------------------
# Encode/decode (run, lumi) as one uint64
# ----------------------------------------------------------------------

def encode_run_lumi(run, lumi):
    """
    Encode (run, lumi) into a single uint64.

    upper 32 bits : run
    lower 32 bits : luminosityBlock
    """
    return (
        (np.asarray(run, dtype=np.uint64) << np.uint64(32))
        | np.asarray(lumi, dtype=np.uint64)
    )


def decode_run_lumi(keys):
    keys = np.asarray(keys, dtype=np.uint64)

    runs = (keys >> np.uint64(32)).astype(np.uint32)
    lumis = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    return runs, lumis


# ----------------------------------------------------------------------
# CMS JSON helpers
# ----------------------------------------------------------------------

def compress_lumis(lumis):
    """
    [1,2,3,5,6,10]
        ->
    [[1,3], [5,6], [10,10]]
    """

    lumis = sorted(set(int(x) for x in lumis))

    if not lumis:
        return []

    ranges = []

    start = lumis[0]
    previous = lumis[0]

    for lumi in lumis[1:]:

        if lumi == previous + 1:
            previous = lumi
            continue

        ranges.append([start, previous])

        start = lumi
        previous = lumi

    ranges.append([start, previous])

    return ranges


def load_certification_keys(filename):
    """
    Read standard CMS certification JSON and return encoded
    (run, lumi) keys as a sorted numpy array.
    """

    with open(filename) as f:
        data = json.load(f)

    chunks = []

    for run_str, ranges in data.items():

        run = int(run_str)

        for first_ls, last_ls in ranges:

            lumis = np.arange(
                first_ls,
                last_ls + 1,
                dtype=np.uint64
            )

            runs = np.full(
                len(lumis),
                run,
                dtype=np.uint64
            )

            chunks.append(
                encode_run_lumi(runs, lumis)
            )

    if not chunks:
        return np.empty(0, dtype=np.uint64)

    return np.unique(
        np.concatenate(chunks)
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract run/luminosityBlock information from "
            "NanoAOD LuminosityBlocks trees and produce a CMS JSON."
        )
    )

    parser.add_argument(
        "input",
        help="Directory containing skimmed NanoAOD ROOT files"
    )

    parser.add_argument(
        "-o", "--output",
        default="processed_lumis.json",
        help="Output CMS JSON"
    )
    
    parser.add_argument(
        "-t", "--tree-name",
        default="LuminosityBlocks",
        help="TTree name to access run and luminosityBlock branches"
    )

    parser.add_argument(
        "--cert-json",
        default=None,
        help=(
            "Optional CMS certification JSON. "
            "Only certified run/lumisections are retained."
        )
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for ROOT files"
    )

    parser.add_argument(
        "--step-size",
        default=100, # "1000 MB"
        help="uproot chunk size, e.g. '50 MB', '100 MB', '500 MB' or number of trees which is integer "
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Find input files
    # ------------------------------------------------------------------

    input_path = Path(args.input)

    if args.recursive:
        files = sorted(input_path.rglob("*.root"))
    else:
        files = sorted(input_path.glob("*.root"))

    if not files:
        raise RuntimeError(
            f"No ROOT files found under {input_path}"
        )

    print(f"Found {len(files)} ROOT files")

    # uproot syntax: file.root:TreeName
    sources = [
        f"{filename}:{args.tree_name}"
        for filename in files
    ]

    # ------------------------------------------------------------------
    # Read all LuminosityBlocks
    #
    # Important:
    #   - no explicit file-by-file uproot.open()
    #   - no Python loop over individual LS entries
    #   - numpy handles deduplication
    # ------------------------------------------------------------------

    unique_chunks = []

    n_entries = 0

    print(f"Reading {args.tree_name} ...")

    iterator = uproot.iterate(
        sources,
        expressions=[
            "run",
            "luminosityBlock",
        ],
        library="np",
        step_size=args.step_size,
    )

    for arrays in tqdm(
        iterator,
        desc=f"Reading {args.tree_name}",
        unit="chunk",
    ):
        runs = arrays["run"]
        lumis = arrays["luminosityBlock"]

        n_entries += len(runs)

        keys = encode_run_lumi(
            runs,
            lumis
        )

        # Deduplicate already inside each chunk.
        #
        # This can significantly reduce memory if the same LS
        # occurs repeatedly across files.
        unique_chunks.append(
            np.unique(keys)
        )

    if not unique_chunks:
        raise RuntimeError(
            f"No {args.tree_name} entries were found"
        )

    # One final global unique operation.
    all_keys = np.unique(
        np.concatenate(unique_chunks)
    )

    n_unique_input = len(all_keys)

    print(
        f"Raw LumiBlock entries : {n_entries}"
    )

    print(
        f"Unique run/LS         : {n_unique_input}"
    )

    # ------------------------------------------------------------------
    # Certification filtering
    # ------------------------------------------------------------------

    if args.cert_json:

        print()
        print(
            f"Loading certification JSON: "
            f"{args.cert_json}"
        )

        cert_keys = load_certification_keys(
            args.cert_json
        )

        print(
            f"Certified run/LS in JSON : {len(cert_keys)}"
        )

        # numpy intersection is much faster than doing:
        #
        #     if lumi in certified[run]
        #
        # for every LS in Python.
        output_keys = np.intersect1d(
            all_keys,
            cert_keys,
            assume_unique=True
        )

        rejected_keys = np.setdiff1d(
            all_keys,
            cert_keys,
            assume_unique=True
        )

    else:

        output_keys = all_keys
        rejected_keys = np.empty(
            0,
            dtype=np.uint64
        )

    # ------------------------------------------------------------------
    # Decode keys
    # ------------------------------------------------------------------

    runs, lumis = decode_run_lumi(
        output_keys
    )

    # ------------------------------------------------------------------
    # Build CMS JSON
    # ------------------------------------------------------------------

    output = {}

    if len(runs):

        # Find boundaries between runs.
        unique_runs, start_indices = np.unique(
            runs,
            return_index=True
        )

        for i, run in enumerate(unique_runs):

            start = start_indices[i]

            if i + 1 < len(start_indices):
                end = start_indices[i + 1]
            else:
                end = len(runs)

            run_lumis = lumis[start:end]

            output[str(int(run))] = compress_lumis(
                run_lumis
            )

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------

    with open(args.output, "w") as fout:

        json.dump(
            output,
            fout,
            indent=2,
            sort_keys=True
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 58)
    print("Summary")
    print("=" * 58)

    print(
        f"ROOT files found             : {len(files)}"
    )

    print(
        f"Raw LumiBlock entries read   : {n_entries}"
    )

    print(
        f"Unique run/LS in NanoAODs    : {n_unique_input}"
    )

    if args.cert_json:

        print(
            f"Certified run/LS retained    : {len(output_keys)}"
        )

        print(
            f"Non-certified run/LS removed : {len(rejected_keys)}"
        )

        if n_unique_input:

            fraction = (
                100.0
                * len(output_keys)
                / n_unique_input
            )

            print(
                f"Fraction retained            : {fraction:.2f}%"
            )

    else:

        print(
            f"Unique run/LS in output      : {len(output_keys)}"
        )

    print(
        f"Runs in output               : {len(output)}"
    )

    print(
        f"Output JSON                  : {args.output}"
    )

    print("=" * 58)


if __name__ == "__main__":
    main()