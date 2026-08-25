from pathlib import Path

import numpy as np

from .runtime import configure_matplotlib_runtime

configure_matplotlib_runtime()

import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .objects import ratio_with_uncertainty
from .plotters import (
    use_hep_style,
    save_figure,
    draw_cms_label,
    auto_range,
    _step_band,
    _mc_ratio_band,
)


# Explicit study-independent fraction colors.
# These can be overridden through component_colors.
DEFAULT_FRACTION_COLORS = {
    "chHEF": "#d62728",   # red
    "neHEF": "#5DA5DA",   # sky blue
    "chEmEF": "#9467bd",  # purple
    "neEmEF": "#f2c744",  # yellow
    "muEF": "#9e9e9e",    # gray
}


def _assert_compatible(
    data_components,
    mc_components,
):
    keys = list(
        data_components.keys()
    )

    if keys != list(
        mc_components.keys()
    ):
        raise ValueError(
            "Data/MC component ordering differs."
        )

    reference = (
        data_components[
            keys[0]
        ]
    )

    for key in keys:
        data = data_components[key]
        mc = mc_components[key]

        if not np.allclose(
            data.edges,
            reference.edges,
        ):
            raise ValueError(
                "Data component binning differs."
            )

        if not np.allclose(
            mc.edges,
            reference.edges,
        ):
            raise ValueError(
                "MC component binning differs."
            )

    return (
        keys,
        reference,
    )


def plot_fraction_composition_data_mc(
    data_components,
    mc_components,
    output,
    *,
    labels=None,
    component_colors=None,
    stack_order="ascending",
    legend_fontsize="x-small",
    xlabel=r"$p_T^Z$ [GeV]",
    ylabel="Energy fraction",
    ratio_label="Data / MC",
    cms_label="Preliminary",
    lumi=None,
    com=13.6,
    method_label=None,
    eta_label=None,
    xlim=None,
    logx=True,
    ratio_ylim=None,
    auto_ratio_y=True,
    ratio_padding=0.12,
    mc_ratio_uncertainty_band=False,
    mc_ratio_band_alpha=0.18,
):
    """
    CMS-style fraction-composition plot.

    Top:
      * MC is a filled histogram stack.
      * Data are colored markers on cumulative fraction boundaries.
      * y range is fixed strictly to [0, 1].

    Bottom:
      * individual fraction Data/MC ratios.
      * optionally, colored MC uncertainty bands around 1.

    Data markers use the same color as the corresponding MC fraction.
    """
    use_hep_style()

    keys, reference = _assert_compatible(
        data_components,
        mc_components,
    )

    x = reference.centers

    colors = dict(
        DEFAULT_FRACTION_COLORS
    )

    if component_colors:
        colors.update(
            component_colors
        )

    # Reserve a real right-side information column so legends can never be
    # clipped by savefig.
    fig = plt.figure(
        figsize=(11.2, 7.7)
    )

    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[
            0.72,
            0.28,
        ],
        width_ratios=[
            1.0,
            0.44,
        ],
        hspace=0.04,
        wspace=0.04,
    )

    ax = fig.add_subplot(
        gs[0, 0]
    )

    rax = fig.add_subplot(
        gs[1, 0],
        sharex=ax,
    )

    side_ax = fig.add_subplot(
        gs[:, 1]
    )
    side_ax.axis("off")

    plt.setp(
        ax.get_xticklabels(),
        visible=False,
    )

    # Optional stack ordering by the average visible MC contribution.
    #
    # ascending:
    #   smallest average contribution first
    #
    # descending:
    #   largest average contribution first
    #
    # input:
    #   preserve the input OrderedDict ordering
    if xlim is None:
        visible = np.ones_like(
            reference.centers,
            dtype=bool,
        )
    else:
        visible = (
            (reference.centers >= xlim[0])
            & (reference.centers <= xlim[1])
        )

    def _average_visible(key):
        values = np.asarray(
            mc_components[key].values,
            dtype=float,
        )

        mask = (
            visible
            & np.isfinite(values)
        )

        if not np.any(mask):
            return 0.0

        return float(
            np.nanmean(
                np.abs(
                    values[mask]
                )
            )
        )

    if stack_order == "ascending":
        keys = sorted(
            keys,
            key=_average_visible,
        )
    elif stack_order == "descending":
        keys = sorted(
            keys,
            key=_average_visible,
            reverse=True,
        )
    elif stack_order == "input":
        keys = list(
            keys
        )
    else:
        raise ValueError(
            "Unknown stack_order '{}'. "
            "Use 'ascending', 'descending', or 'input'."
            .format(
                stack_order
            )
        )

    display_labels = [
        (
            labels.get(
                key,
                key,
            )
            if labels
            else key
        )
        for key in keys
    ]

    mc_values = [
        np.asarray(
            mc_components[key].values,
            dtype=float,
        )
        for key in keys
    ]

    data_values = [
        np.asarray(
            data_components[key].values,
            dtype=float,
        )
        for key in keys
    ]

    component_color_list = [
        colors.get(
            key,
            None,
        )
        for key in keys
    ]

    # ------------------------------------------------------------
    # Filled MC composition as real histogram bins.
    # ------------------------------------------------------------
    hep.histplot(
        [
            (
                values,
                reference.edges,
            )
            for values in mc_values
        ],
        stack=True,
        histtype="fill",
        color=component_color_list,
        alpha=0.90,
        edgecolor="black",
        linewidth=0.45,
        ax=ax,
    )

    # ------------------------------------------------------------
    # Data cumulative boundaries with component-matched colors.
    # ------------------------------------------------------------
    cumulative_data = np.cumsum(
        np.asarray(
            data_values
        ),
        axis=0,
    )

    cumulative_data_error2 = np.zeros_like(
        cumulative_data
    )

    running_error2 = np.zeros_like(
        data_values[0]
    )

    for index, key in enumerate(
        keys
    ):
        error = (
            np.zeros_like(
                data_values[index]
            )
            if data_components[key].errors is None
            else np.asarray(
                data_components[key].errors,
                dtype=float,
            )
        )

        running_error2 = (
            running_error2
            + error ** 2
        )

        cumulative_data_error2[
            index
        ] = running_error2

    for index, values in enumerate(
        cumulative_data
    ):
        ax.errorbar(
            x,
            values,
            yerr=np.sqrt(
                cumulative_data_error2[
                    index
                ]
            ),
            fmt="o",
            linestyle="none",
            markersize=3.0,
            linewidth=0.75,
            color=component_color_list[
                index
            ],
            markeredgecolor="black",
            markeredgewidth=0.25,
            capsize=0,
            zorder=4,
        )

    # ------------------------------------------------------------
    # Lower panel: individual Data/MC ratios.
    # ------------------------------------------------------------
    ratio_series = []

    for index, key in enumerate(
        keys
    ):
        data = data_components[key]
        mc = mc_components[key]

        denominator_error_for_points = (
            None
            if mc_ratio_uncertainty_band
            else mc.errors
        )

        ratio, ratio_error = (
            ratio_with_uncertainty(
                data.values,
                mc.values,
                data.errors,
                denominator_error_for_points,
            )
        )

        mask = np.isfinite(
            ratio
        )

        if ratio_error is not None:
            mask = (
                mask
                & np.isfinite(
                    ratio_error
                )
            )

        if xlim is not None:
            mask = (
                mask
                & (
                    x >= xlim[0]
                )
                & (
                    x <= xlim[1]
                )
            )

        rax.errorbar(
            x,
            ratio,
            yerr=ratio_error,
            fmt="o",
            linestyle="none",
            markersize=3.0,
            linewidth=0.85,
            color=component_color_list[
                index
            ],
            label=display_labels[index],
            zorder=4,
        )

        ratio_series.append(
            (
                ratio,
                ratio_error,
                mask,
            )
        )

        if mc_ratio_uncertainty_band:
            band = _mc_ratio_band(
                mc.values,
                mc.errors,
            )

            if band is not None:
                band_low, band_high, band_valid = band

                _step_band(
                    rax,
                    mc.edges,
                    band_low,
                    band_high,
                    color=component_color_list[
                        index
                    ],
                    alpha=mc_ratio_band_alpha,
                    zorder=1,
                )

                band_mask = (
                    band_valid.copy()
                )

                if xlim is not None:
                    band_mask = (
                        band_mask
                        & (
                            x >= xlim[0]
                        )
                        & (
                            x <= xlim[1]
                        )
                    )

                ratio_series.append(
                    (
                        np.ones_like(
                            mc.values
                        ),
                        np.where(
                            band_mask,
                            0.5
                            * (
                                band_high
                                - band_low
                            ),
                            np.nan,
                        ),
                        band_mask,
                    )
                )

    rax.axhline(
        1.0,
        linewidth=1,
        linestyle="--",
        color="black",
    )

    # ------------------------------------------------------------
    # Axes.
    # ------------------------------------------------------------
    if xlim is not None:
        ax.set_xlim(
            *xlim
        )

    if logx:
        ax.set_xscale(
            "log"
        )

    # Strictly [0,1], as requested.
    ax.set_ylim(
        0.0,
        1.0,
    )

    if ratio_ylim is not None:
        rax.set_ylim(
            *ratio_ylim
        )
    elif auto_ratio_y:
        rax.set_ylim(
            *auto_range(
                ratio_series,
                include_errors=True,
                padding=ratio_padding,
                reference=1.0,
                minimum_span=0.05,
            )
        )

    ax.set_ylabel(
        ylabel
    )

    rax.set_ylabel(
        ratio_label
    )

    rax.set_xlabel(
        xlabel
    )

    # ------------------------------------------------------------
    # Put legend + explanatory text outside the plotting axes.
    # ------------------------------------------------------------
    component_handles = [
        Patch(
            facecolor=component_color_list[
                index
            ],
            edgecolor="black",
            linewidth=0.4,
            label=display_labels[
                index
            ],
        )
        for index in range(
            len(keys)
        )
    ]

    style_handles = [
        Patch(
            facecolor="0.75",
            edgecolor="black",
            label="MC: filled histogram",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="black",
            markersize=5,
            label="Data: markers",
        ),
    ]

    side_ax.legend(
        handles=(
            component_handles
            + style_handles
        ),
        loc="upper left",
        bbox_to_anchor=(
            0.02,
            0.98,
        ),
        borderaxespad=0.0,
        frameon=False,
        fontsize=legend_fontsize,
    )

    side_lines = []

    if eta_label:
        side_lines.append(
            eta_label
        )

    if method_label:
        side_lines.append(
            method_label
        )

    if side_lines:
        side_ax.text(
            0.02,
            0.42,
            "\n".join(
                side_lines
            ),
            transform=side_ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
        )

    draw_cms_label(
        ax,
        cms_label=cms_label,
        lumi=lumi,
        com=com,
        data=True,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.12,
        top=0.94,
    )

    save_figure(
        fig,
        output,
    )
