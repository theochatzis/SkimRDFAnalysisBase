from .runtime import configure_matplotlib_runtime

configure_matplotlib_runtime()

from .objects import (
    Hist1D,
    Hist2D,
    Hist3D,
    Graph1D,
    ratio_with_uncertainty,
)

from .rootio import (
    RootFileReader,
    read_root_object,
)

from .plotters import (
    plot_hist1d_data_mc,
    plot_hist1d_methods_data_mc,
    save_figure,
    auto_range,
)

from .composition import (
    plot_fraction_composition_data_mc,
)

__all__ = [
    "Hist1D",
    "Hist2D",
    "Hist3D",
    "Graph1D",
    "ratio_with_uncertainty",
    "RootFileReader",
    "read_root_object",
    "plot_hist1d_data_mc",
    "plot_hist1d_methods_data_mc",
    "plot_fraction_composition_data_mc",
    "save_figure",
    "auto_range",
]
