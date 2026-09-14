import os
import re

import numpy as np
import ROOT
import yaml


_VALID_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$"
)


def load_weights_config(path, resolve_file):
    """Load and validate the YAML weight configuration."""
    if not path:
        return {}, ""

    path = resolve_file(
        path,
        "Weight configuration"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle) or {}

    weights = config.get(
        "weights"
    )

    if not isinstance(weights, dict):
        raise TypeError(
            f"Weight configuration '{path}' needs "
            "a top-level 'weights:' mapping."
        )

    config["_config_dir"] = os.path.dirname(path)

    validate_weights_config(
        config
    )

    return config, path


def validate_weights_config(config):
    event_weight_name = str(
        config.get(
            "event_weight_name",
            "eventWeight"
        )
    )

    if not _VALID_NAME.match(event_weight_name):
        raise ValueError(
            f"Invalid event_weight_name "
            f"'{event_weight_name}'."
        )

    for name, spec in config.get(
        "weights",
        {}
    ).items():
        if not _VALID_NAME.match(name):
            raise ValueError(
                f"Invalid weight name '{name}'."
            )

        if name == event_weight_name:
            raise ValueError(
                f"Weight '{name}' conflicts with "
                "event_weight_name."
            )

        if not isinstance(spec, dict):
            raise TypeError(
                f"Weight '{name}' must be a dictionary."
            )

        weight_type = spec.get(
            "type",
            "expression"
        )

        if weight_type not in (
            "expression",
            "correctionlib",
        ):
            raise ValueError(
                f"Unsupported weight type "
                f"'{weight_type}' for '{name}'."
            )

        apply_to = spec.get(
            "apply_to",
            "mc"
        )

        if apply_to not in (
            "all",
            "mc",
            "data",
        ):
            raise ValueError(
                f"Weight '{name}' apply_to must "
                "be all, mc, or data."
            )

        if weight_type == "expression":
            expressions = (
                spec.get(
                    "expressions",
                    {}
                )
                or {}
            )

            if "nominal" not in expressions:
                raise RuntimeError(
                    f"Expression weight '{name}' requires "
                    "expressions.nominal."
                )

        if weight_type == "correctionlib":
            for field in (
                "json",
                "correction",
                "arguments",
                "variations",
            ):
                if field not in spec:
                    raise RuntimeError(
                        f"Correctionlib weight '{name}' "
                        f"requires '{field}'."
                    )

            variations = (
                spec.get(
                    "variations",
                    {}
                )
                or {}
            )

            if "nominal" not in variations:
                raise RuntimeError(
                    f"Correctionlib weight '{name}' requires "
                    "variations.nominal."
                )

        histogram = spec.get(
            "histogram"
        )

        if histogram is not None:
            if not isinstance(histogram, dict):
                raise TypeError(
                    f"Weight '{name}' histogram "
                    "must be a dictionary."
                )

            if (
                "edges" not in histogram
                and "bins" not in histogram
            ):
                raise RuntimeError(
                    f"Weight '{name}' histogram needs "
                    "'edges' or 'bins'."
                )


def annotate_sample_type(dataframe, sample):
    """
    Detect standard NanoAOD MC from genWeight.

    The result is saved in sample["is_mc"] / sample["is_data"] so the
    analysis definition can also use it after this point.
    """
    columns = {
        str(name)
        for name in dataframe.GetColumnNames()
    }

    sample["is_mc"] = (
        "genWeight" in columns
    )
    sample["is_data"] = (
        not sample["is_mc"]
    )

    return sample


def _resolve_payload_path(path, config):
    path = os.path.expanduser(
        str(path)
    )

    if os.path.isabs(path):
        return path

    if os.path.isfile(path):
        return os.path.abspath(path)

    config_dir = config.get(
        "_config_dir",
        ""
    )

    if config_dir:
        path = os.path.join(
            config_dir,
            path
        )

    return os.path.abspath(path)


def setup_weight_corrections(config, repo_dir):
    """
    Register every correctionlib-backed weight once before the sample loop.
    """
    correction_weights = {
        name: spec
        for name, spec in config.get(
            "weights",
            {}
        ).items()
        if (
            spec.get("enabled", True)
            and spec.get("type", "expression") == "correctionlib"
        )
    }

    if not correction_weights:
        return

    common_dir = os.path.join(
        repo_dir,
        "Common"
    )
    interface_dir = os.path.join(
        common_dir,
        "interface"
    )
    library = os.path.join(
        common_dir,
        "libAnalysisCommon.so"
    )

    if not os.path.isfile(library):
        raise RuntimeError(
            f"Correctionlib weights need '{library}'.\n"
            "Compile first with:\n"
            "  bash setup.sh"
        )

    import correctionlib

    ROOT.gInterpreter.AddIncludePath(os.path.join(correctionlib.__path__[0], "include"))
    ROOT.gInterpreter.AddIncludePath(interface_dir)
    if not ROOT.gInterpreter.Declare('#include "CorrectionRegistry.h"'):
        raise RuntimeError("Could not declare CorrectionRegistry.h")

    status = ROOT.gSystem.Load(
        library
    )

    if status < 0:
        raise RuntimeError(
            f"Could not load correction library: "
            f"{library}"
        )

    for name, spec in correction_weights.items():
        json_path = _resolve_payload_path(
            spec["json"],
            config
        )

        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"Weight '{name}' JSON does not exist: "
                f"{json_path}"
            )

        ROOT.skimrdf.registerCorrection(
            f"event_weight_{name}",
            json_path,
            str(spec["correction"]),
            bool(
                spec.get(
                    "compound",
                    False
                )
            ),
        )

        print(
            f"[weights] Registered '{name}' -> "
            f"{spec['correction']}"
        )


def _sample_rule_matches(sample_name, spec):
    rules = (
        spec.get(
            "samples",
            {}
        )
        or {}
    )

    include = (
        rules.get(
            "include",
            []
        )
        or []
    )
    exclude = (
        rules.get(
            "exclude",
            []
        )
        or []
    )

    if isinstance(include, str):
        include = [include]

    if isinstance(exclude, str):
        exclude = [exclude]

    if (
        include
        and not any(
            re.search(pattern, sample_name)
            for pattern in include
        )
    ):
        return False

    if (
        exclude
        and any(
            re.search(pattern, sample_name)
            for pattern in exclude
        )
    ):
        return False

    return True


def _applies_to_sample(spec, sample):
    apply_to = spec.get(
        "apply_to",
        "mc"
    )

    if (
        apply_to == "mc"
        and not sample.get(
            "is_mc",
            False
        )
    ):
        return False

    if (
        apply_to == "data"
        and not sample.get(
            "is_data",
            False
        )
    ):
        return False

    return _sample_rule_matches(
        sample.get(
            "name",
            ""
        ),
        spec
    )


def _variation_source(spec):
    if spec.get(
        "type",
        "expression"
    ) == "expression":
        return (
            spec.get(
                "expressions",
                {}
            )
            or {}
        )

    return (
        spec.get(
            "variations",
            {}
        )
        or {}
    )


def _required_columns(spec):
    required = set(
        spec.get(
            "requires",
            []
        )
        or []
    )

    if spec.get(
        "type",
        "expression"
    ) == "correctionlib":
        for argument in spec.get(
            "arguments",
            []
        ):
            if isinstance(argument, str):
                required.add(
                    argument
                )

            elif (
                isinstance(argument, dict)
                and "column" in argument
            ):
                required.add(
                    str(
                        argument["column"]
                    )
                )

    return required


def _check_required_columns(dataframe, name, spec):
    required = _required_columns(
        spec
    )

    if not required:
        return

    columns = {
        str(column)
        for column in dataframe.GetColumnNames()
    }

    missing = sorted(
        required - columns
    )

    if missing:
        raise RuntimeError(
            f"Weight '{name}' requires missing RDF column(s): "
            f"{', '.join(missing)}"
        )


def _cpp_string(value):
    value = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
    )

    return f'std::string("{value}")'


def _cpp_argument(argument, variation_value=None):
    """
    Convert one YAML correctionlib argument into a C++ RDF expression.

    Supported examples:
      - column: Pileup_nTrueInt
        type: real

      - variation: true
        type: string

      - value: 123
        type: int
    """
    if isinstance(argument, str):
        return (
            f"static_cast<double>"
            f"({argument})"
        )

    if not isinstance(argument, dict):
        raise TypeError(
            "Correctionlib arguments must be "
            "strings or dictionaries."
        )

    arg_type = argument.get(
        "type",
        "real"
    )

    if argument.get(
        "variation",
        False
    ):
        if variation_value is None:
            raise RuntimeError(
                "variation: true requires "
                "a variation value."
            )

        value = variation_value
        source = "value"

    elif "column" in argument:
        value = argument["column"]
        source = "column"

    elif "value" in argument:
        value = argument["value"]
        source = "value"

    else:
        raise RuntimeError(
            "Correctionlib argument needs "
            "column, value, or variation."
        )

    if arg_type == "string":
        if source == "column":
            return f"std::string({value})"

        return _cpp_string(
            value
        )

    if arg_type == "int":
        if source == "column":
            return (
                f"static_cast<int64_t>"
                f"({value})"
            )

        return (
            f"static_cast<int64_t>"
            f"({int(value)})"
        )

    if arg_type == "real":
        if source == "column":
            return (
                f"static_cast<double>"
                f"({value})"
            )

        return repr(
            float(value)
        )

    raise ValueError(
        f"Unsupported correctionlib argument "
        f"type '{arg_type}'."
    )


def _correction_expression(name, spec, variation_value):
    arguments = ", ".join(
        _cpp_argument(
            argument,
            variation_value
        )
        for argument in spec["arguments"]
    )

    return (
        f'skimrdf::evaluateCorrection('
        f'"event_weight_{name}", '
        f'std::vector<correction::Variable::Type>'
        f'{{{arguments}}})'
    )


def _weight_expressions(name, spec):
    if spec.get(
        "type",
        "expression"
    ) == "expression":
        return {
            key: str(value)
            for key, value in spec[
                "expressions"
            ].items()
            if key in (
                "nominal",
                "up",
                "down",
            )
        }

    variations = spec[
        "variations"
    ]

    return {
        key: _correction_expression(
            name,
            spec,
            variations[key]
        )
        for key in (
            "nominal",
            "up",
            "down",
        )
        if key in variations
    }


def apply_event_weights(dataframe, config, sample):
    """
    Define individual weights, variations, and combined event weights.

    For weights named:
      puWeight
      muonWeight

    the following columns are created when variations exist:
      puWeight
      puWeight_up
      puWeight_down
      muonWeight
      muonWeight_up
      muonWeight_down

      eventWeight
      eventWeight_puWeight_up
      eventWeight_puWeight_down
      eventWeight_muonWeight_up
      eventWeight_muonWeight_down

    A combined systematic column varies one weight and keeps all other
    configured weights nominal.
    """
    weights = {
        name: spec
        for name, spec in config.get(
            "weights",
            {}
        ).items()
        if spec.get(
            "enabled",
            True
        )
    }

    if not weights:
        return dataframe, {
            "enabled": False,
            "event_weight": None,
            "configured": [],
            "active": [],
            "variations": {},
        }

    event_weight = str(
        config.get(
            "event_weight_name",
            "eventWeight"
        )
    )

    input_columns = {
        str(name)
        for name in dataframe.GetColumnNames()
    }

    reserved = {
        event_weight
    }

    for name, spec in weights.items():
        reserved.add(
            name
        )

        variation_source = _variation_source(
            spec
        )

        for direction in (
            "up",
            "down",
        ):
            if direction not in variation_source:
                continue

            reserved.add(
                f"{name}_{direction}"
            )
            reserved.add(
                f"{event_weight}_{name}_{direction}"
            )

    collisions = sorted(
        input_columns & reserved
    )

    if collisions:
        raise RuntimeError(
            "Weight system would overwrite RDF column(s): "
            + ", ".join(collisions)
        )

    names = list(
        weights
    )
    active = []

    for name, spec in weights.items():
        applies = _applies_to_sample(
            spec,
            sample
        )

        variation_source = _variation_source(
            spec
        )

        if applies:
            _check_required_columns(
                dataframe,
                name,
                spec
            )

            expressions = _weight_expressions(
                name,
                spec
            )
            active.append(
                name
            )

        else:
            expressions = {
                "nominal": "1.0"
            }

            for direction in (
                "up",
                "down",
            ):
                if direction in variation_source:
                    expressions[
                        direction
                    ] = "1.0"

        dataframe = dataframe.Define(
            name,
            expressions["nominal"]
        )

        for direction in (
            "up",
            "down",
        ):
            if direction not in expressions:
                continue

            dataframe = dataframe.Define(
                f"{name}_{direction}",
                expressions[direction]
            )

    dataframe = dataframe.Define(
        event_weight,
        " * ".join(names)
    )

    combined_variations = {}

    for name, spec in weights.items():
        variation_source = _variation_source(
            spec
        )

        for direction in (
            "up",
            "down",
        ):
            if direction not in variation_source:
                continue

            varied_terms = [
                (
                    f"{name}_{direction}"
                    if other == name
                    else other
                )
                for other in names
            ]

            column = (
                f"{event_weight}_"
                f"{name}_{direction}"
            )

            dataframe = dataframe.Define(
                column,
                " * ".join(
                    varied_terms
                )
            )

            combined_variations[
                f"{name}_{direction}"
            ] = column

    print(
        f"[weights:{sample.get('name', '<sample>')}] "
        f"active: "
        f"{', '.join(active) if active else 'none'}"
    )

    return dataframe, {
        "enabled": True,
        "event_weight": event_weight,
        "configured": names,
        "active": active,
        "variations": combined_variations,
    }


def _make_edges(spec):
    if (
        len(spec) > 0
        and spec[0] == "bins"
    ):
        return np.linspace(
            float(spec[2]),
            float(spec[3]),
            int(spec[1]) + 1,
            dtype=np.float64
        )

    return np.asarray(
        spec,
        dtype=np.float64
    )


def _hist_model(name, histogram):
    title = histogram.get(
        "title",
        f"{name};{name};Events"
    )

    if "edges" in histogram:
        edges = _make_edges(
            histogram["edges"]
        )

        return (
            name,
            title,
            len(edges) - 1,
            edges
        )

    bins = histogram[
        "bins"
    ]

    return (
        name,
        title,
        int(bins[0]),
        float(bins[1]),
        float(bins[2])
    )


def book_weight_histograms(dataframe, config, state):
    """
    Book unweighted monitoring histograms for each active weight.

    By default nominal/up/down distributions are all stored when the
    variations exist. Set include_variations: false to keep nominal only.
    """
    actions = []

    for name in state.get(
        "active",
        []
    ):
        spec = config[
            "weights"
        ][name]

        histogram = spec.get(
            "histogram"
        )

        if not histogram:
            continue

        model = _hist_model(
            name,
            histogram
        )

        actions.append(
            dataframe.Histo1D(
                model,
                name
            )
        )

        if not histogram.get(
            "include_variations",
            True
        ):
            continue

        variation_source = _variation_source(
            spec
        )

        for direction in (
            "up",
            "down",
        ):
            if direction not in variation_source:
                continue

            column = (
                f"{name}_{direction}"
            )

            varied_model = list(
                model
            )
            varied_model[0] = column
            varied_model[1] = histogram.get(
                f"title_{direction}",
                f"{name} {direction};{name};Events"
            )

            actions.append(
                dataframe.Histo1D(
                    tuple(varied_model),
                    column
                )
            )

    return actions
