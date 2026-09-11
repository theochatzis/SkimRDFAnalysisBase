from collections import OrderedDict

import numpy as np

from .objects import (
    Graph1D,
    Hist1D,
    ratio_with_uncertainty,
)
from .plotters import (
    _mc_ratio_band,
    _step_band,
    draw_cms_label,
    ratio_axes,
    save_figure,
    use_hep_style,
)

import matplotlib.pyplot as plt


def _ordered_mapping(objects, default_key="value"):
    if isinstance(objects, (Hist1D, Graph1D)):
        return OrderedDict([
            (default_key, objects),
        ])

    return OrderedDict(objects)


def _check_same_binning(histograms):
    histograms = list(histograms)

    if not histograms:
        raise ValueError("At least one histogram is required.")

    reference = histograms[0].edges

    for histogram in histograms[1:]:
        if not np.allclose(
            histogram.edges,
            reference,
        ):
            raise ValueError(
                "Histogram binning differs between inputs."
            )


def sum_histograms(
    histograms,
    *,
    name="sum",
    title="",
):
    """
    Sum Hist1D objects bin-by-bin.

    Statistical uncertainties are added in quadrature. If none of the input
    histograms carries uncertainties, the returned histogram has errors=None.
    """
    histograms = list(histograms)

    _check_same_binning(
        histograms
    )

    values = np.sum(
        [
            histogram.values
            for histogram in histograms
        ],
        axis=0,
    )

    have_errors = any(
        histogram.errors is not None
        for histogram in histograms
    )

    if have_errors:
        variance = np.zeros_like(
            values,
            dtype=float,
        )

        for histogram in histograms:
            if histogram.errors is None:
                continue

            variance += (
                np.asarray(
                    histogram.errors,
                    dtype=float,
                )
                ** 2
            )

        errors = np.sqrt(
            variance
        )

    else:
        errors = None

    reference = histograms[0]

    return Hist1D(
        values=values,
        edges=reference.edges.copy(),
        errors=errors,
        name=name,
        title=title,
        xlabel=reference.xlabel,
        ylabel=reference.ylabel,
    )


def hist_to_graph(
    histogram,
    *,
    include_x_errors=True,
    drop_empty=False,
    name=None,
):
    """
    Convert a binned Hist1D/TProfile representation to Graph1D.

    This is useful for plotting profile-like ROOT objects with the generic
    graph plotter.
    """
    x = histogram.centers.copy()
    y = histogram.values.copy()

    if include_x_errors:
        xerr_low = 0.5 * histogram.widths
        xerr_high = 0.5 * histogram.widths
    else:
        xerr_low = None
        xerr_high = None

    if histogram.errors is None:
        yerr_low = None
        yerr_high = None
    else:
        yerr_low = histogram.errors.copy()
        yerr_high = histogram.errors.copy()

    if drop_empty:
        mask = np.isfinite(y)

        if histogram.errors is not None:
            mask &= np.isfinite(
                histogram.errors
            )
            mask &= (
                (y != 0.0)
                | (histogram.errors != 0.0)
            )
        else:
            mask &= (
                y != 0.0
            )

        x = x[mask]
        y = y[mask]

        if xerr_low is not None:
            xerr_low = xerr_low[mask]
            xerr_high = xerr_high[mask]

        if yerr_low is not None:
            yerr_low = yerr_low[mask]
            yerr_high = yerr_high[mask]

    return Graph1D(
        x=x,
        y=y,
        xerr_low=xerr_low,
        xerr_high=xerr_high,
        yerr_low=yerr_low,
        yerr_high=yerr_high,
        name=(
            histogram.name
            if name is None
            else name
        ),
        title=histogram.title,
        xlabel=histogram.xlabel,
        ylabel=histogram.ylabel,
    )


def efficiency_graph(
    numerator,
    denominator,
    *,
    confidence_level=0.682689492137,
    interval="clopper-pearson",
    include_x_errors=True,
    drop_empty=True,
    name="efficiency",
    title="",
):
    """
    Build a Graph1D from numerator/denominator Hist1D objects.

    Supported intervals:
      * "clopper-pearson" (default): ROOT TEfficiency exact binomial interval.
        Numerator and denominator must contain unweighted integer counts.
      * "normal": symmetric sqrt(e(1-e)/N) approximation.

    Bins with denominator <= 0 are omitted when drop_empty=True.
    """
    _check_same_binning([
        numerator,
        denominator,
    ])

    interval = interval.lower()

    if interval not in (
        "clopper-pearson",
        "normal",
    ):
        raise ValueError(
            "interval must be 'clopper-pearson' or 'normal'."
        )

    x_values = []
    y_values = []
    xerr_low_values = []
    xerr_high_values = []
    yerr_low_values = []
    yerr_high_values = []

    if interval == "clopper-pearson":
        import ROOT

    for index, (
        passed_value,
        total_value,
    ) in enumerate(zip(
        numerator.values,
        denominator.values,
    )):
        if (
            not np.isfinite(passed_value)
            or not np.isfinite(total_value)
            or total_value <= 0.0
        ):
            if drop_empty:
                continue

            efficiency = np.nan
            error_low = np.nan
            error_high = np.nan

        else:
            if (
                passed_value < 0.0
                or passed_value > total_value
            ):
                raise ValueError(
                    "Efficiency numerator must satisfy "
                    "0 <= numerator <= denominator in every bin."
                )

            efficiency = (
                passed_value
                / total_value
            )

            if interval == "clopper-pearson":
                passed_integer = int(
                    round(passed_value)
                )
                total_integer = int(
                    round(total_value)
                )

                if (
                    not np.isclose(
                        passed_value,
                        passed_integer,
                        atol=1e-6,
                    )
                    or not np.isclose(
                        total_value,
                        total_integer,
                        atol=1e-6,
                    )
                ):
                    raise ValueError(
                        "Clopper-Pearson intervals require unweighted "
                        "integer numerator/denominator counts. "
                        "Use interval='normal' for weighted histograms."
                    )

                lower = ROOT.TEfficiency.ClopperPearson(
                    total_integer,
                    passed_integer,
                    confidence_level,
                    False,
                )

                upper = ROOT.TEfficiency.ClopperPearson(
                    total_integer,
                    passed_integer,
                    confidence_level,
                    True,
                )

                error_low = (
                    efficiency
                    - lower
                )

                error_high = (
                    upper
                    - efficiency
                )

            else:
                error = np.sqrt(
                    max(
                        efficiency
                        * (
                            1.0
                            - efficiency
                        )
                        / total_value,
                        0.0,
                    )
                )

                error_low = min(
                    error,
                    efficiency,
                )

                error_high = min(
                    error,
                    1.0 - efficiency,
                )

        x_values.append(
            numerator.centers[index]
        )
        y_values.append(
            efficiency
        )

        if include_x_errors:
            half_width = (
                0.5
                * numerator.widths[index]
            )
        else:
            half_width = 0.0

        xerr_low_values.append(
            half_width
        )
        xerr_high_values.append(
            half_width
        )
        yerr_low_values.append(
            error_low
        )
        yerr_high_values.append(
            error_high
        )

    return Graph1D(
        x=np.asarray(
            x_values,
            dtype=float,
        ),
        y=np.asarray(
            y_values,
            dtype=float,
        ),
        xerr_low=np.asarray(
            xerr_low_values,
            dtype=float,
        ),
        xerr_high=np.asarray(
            xerr_high_values,
            dtype=float,
        ),
        yerr_low=np.asarray(
            yerr_low_values,
            dtype=float,
        ),
        yerr_high=np.asarray(
            yerr_high_values,
            dtype=float,
        ),
        name=name,
        title=title,
        xlabel=numerator.xlabel,
        ylabel="Efficiency",
    )


def plot_graphs(
    graphs,
    output,
    *,
    labels=None,
    xlabel="",
    ylabel="",
    xlim=None,
    ylim=None,
    logx=False,
    logy=False,
    connect=False,
    reference_line=None,
    legend_outside=False,
    legend_fontsize="small",
    side_text=None,
    cms_label="Preliminary",
    lumi=None,
    com=13.6,
    data=False,
    title=None,
):
    """
    Generic Graph1D plotter.

    Typical uses:
      * efficiencies and purities;
      * scale factors;
      * TProfiles converted with hist_to_graph();
      * response curves.

    `graphs` can be a Graph1D or an ordered mapping {key: Graph1D}.
    """
    use_hep_style()

    graphs = _ordered_mapping(
        graphs
    )

    if legend_outside:
        fig = plt.figure(
            figsize=(10.8, 7.2)
        )

        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[
                1.0,
                0.38,
            ],
            wspace=0.04,
        )

        ax = fig.add_subplot(
            gs[0, 0]
        )

        side_ax = fig.add_subplot(
            gs[0, 1]
        )
        side_ax.axis("off")

    else:
        fig, ax = plt.subplots(
            figsize=(8.2, 7.2)
        )
        side_ax = None

    plotted_y = []
    plotted_low = []
    plotted_high = []

    for key, graph in graphs.items():
        label = (
            labels.get(
                key,
                key,
            )
            if labels
            else key
        )

        mask = (
            np.isfinite(graph.x)
            & np.isfinite(graph.y)
        )

        if not np.any(mask):
            continue

        x = graph.x[mask]
        y = graph.y[mask]

        if graph.xerr is None:
            xerr = None
        else:
            xerr = graph.xerr[:, mask]

        if graph.yerr is None:
            yerr = None
            ylow = y
            yhigh = y
        else:
            yerr = graph.yerr[:, mask]
            ylow = (
                y
                - yerr[0]
            )
            yhigh = (
                y
                + yerr[1]
            )

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            linestyle=(
                "-"
                if connect
                else "none"
            ),
            markersize=4,
            linewidth=1.5,
            label=label,
        )

        plotted_y.append(
            y
        )
        plotted_low.append(
            ylow
        )
        plotted_high.append(
            yhigh
        )

    if reference_line is not None:
        ax.axhline(
            reference_line,
            linewidth=1,
            linestyle="--",
        )

    if xlim is not None:
        ax.set_xlim(
            *xlim
        )

    if ylim is not None:
        ax.set_ylim(
            *ylim
        )

    elif plotted_y:
        low = min(
            np.nanmin(values)
            for values in plotted_low
        )

        high = max(
            np.nanmax(values)
            for values in plotted_high
        )

        if logy:
            positive = [
                values[
                    values > 0.0
                ]
                for values in plotted_low
                if np.any(
                    values > 0.0
                )
            ]

            if positive:
                ymin = min(
                    np.min(values)
                    for values in positive
                )
            else:
                ymin = 1e-3

            ymax = max(
                high,
                ymin * 10.0,
            )

            ax.set_ylim(
                ymin * 0.7,
                ymax * 1.5,
            )

        else:
            span = max(
                high - low,
                1e-6,
            )

            ax.set_ylim(
                low - 0.10 * span,
                high + 0.15 * span,
            )

    if logx:
        ax.set_xscale(
            "log"
        )

    if logy:
        ax.set_yscale(
            "log"
        )

    if xlabel:
        ax.set_xlabel(
            xlabel
        )

    if ylabel:
        ax.set_ylabel(
            ylabel
        )

    if title:
        ax.text(
            0.03,
            0.91,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
        )

    if legend_outside:
        handles, legend_labels = (
            ax.get_legend_handles_labels()
        )

        side_ax.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(
                0.02,
                0.98,
            ),
            borderaxespad=0.0,
            frameon=False,
            fontsize=legend_fontsize,
        )

        if side_text:
            side_ax.text(
                0.02,
                0.60,
                str(side_text),
                transform=side_ax.transAxes,
                ha="left",
                va="top",
                fontsize="small",
            )

    else:
        ax.legend(
            frameon=False,
            fontsize=legend_fontsize,
        )

    draw_cms_label(
        ax,
        cms_label=cms_label,
        lumi=lumi,
        com=com,
        data=data,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.12,
        top=0.94,
    )

    save_figure(
        fig,
        output,
    )


def plot_efficiency(
    numerator,
    denominator,
    output,
    *,
    label="Efficiency",
    xlabel="",
    ylabel="Efficiency",
    interval="clopper-pearson",
    confidence_level=0.682689492137,
    xlim=None,
    ylim=(0.0, 1.05),
    logx=False,
    cms_label="Simulation",
    lumi=None,
    com=13.6,
    title=None,
):
    """
    Convenience wrapper for one numerator/denominator efficiency curve.
    """
    graph = efficiency_graph(
        numerator,
        denominator,
        confidence_level=confidence_level,
        interval=interval,
    )

    plot_graphs(
        OrderedDict([
            ("efficiency", graph),
        ]),
        output,
        labels={
            "efficiency": label,
        },
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        logx=logx,
        cms_label=cms_label,
        lumi=lumi,
        com=com,
        data=False,
        title=title,
    )


def plot_hist1d_data_mc_stack(
    data,
    mc_components,
    output,
    *,
    labels=None,
    scales=None,
    data_label="Data",
    xlabel="",
    ylabel="Events",
    ratio_label="Data / MC",
    normalize_mc_to_data=False,
    xlim=None,
    ylim=None,
    ratio_ylim=(0.5, 1.5),
    logx=False,
    logy=False,
    mc_uncertainty_band=True,
    mc_band_alpha=0.25,
    legend_outside=False,
    legend_fontsize="small",
    side_text=None,
    cms_label="Preliminary",
    lumi=None,
    com=13.6,
    title=None,
):
    """
    Data versus a stack of MC process Hist1D objects.

    Parameters
    ----------
    data:
        Hist1D containing data yields.

    mc_components:
        Ordered mapping:
            {
                "DY": Hist1D(...),
                "TT": Hist1D(...),
                ...
            }

        The insertion order is the stack order.

    scales:
        Optional mapping of per-process multiplicative scale factors.
        This is useful when the ROOT histograms have not already been
        normalized to luminosity.

    Ratio:
        Data / sum(MC components).

    MC uncertainties:
        Component statistical uncertainties are added in quadrature and shown
        as a band on the total MC and around one in the ratio panel.
    """
    use_hep_style()

    mc_components = OrderedDict(
        mc_components
    )

    if not mc_components:
        raise ValueError(
            "mc_components cannot be empty."
        )

    _check_same_binning(
        [data]
        + list(
            mc_components.values()
        )
    )

    scales = (
        {}
        if scales is None
        else scales
    )

    scaled_components = OrderedDict()

    for key, histogram in mc_components.items():
        factor = float(
            scales.get(
                key,
                1.0,
            )
        )

        scaled_components[key] = (
            histogram.scaled(
                factor
            )
        )

    total_mc = sum_histograms(
        scaled_components.values(),
        name="total_mc",
    )

    if normalize_mc_to_data:
        data_integral = (
            data.integral
        )
        mc_integral = (
            total_mc.integral
        )

        if (
            np.isfinite(data_integral)
            and np.isfinite(mc_integral)
            and mc_integral != 0.0
        ):
            global_scale = (
                data_integral
                / mc_integral
            )

            scaled_components = OrderedDict([
                (
                    key,
                    histogram.scaled(
                        global_scale
                    ),
                )
                for key, histogram
                in scaled_components.items()
            ])

            total_mc = sum_histograms(
                scaled_components.values(),
                name="total_mc",
            )

    fig, ax, rax, side_ax = ratio_axes(
        figsize=(
            (10.8, 7.6)
            if legend_outside
            else (8.2, 7.6)
        ),
        side_panel=legend_outside,
        side_width=0.42,
    )

    edges = data.edges
    widths = data.widths
    left = edges[:-1]

    positive_bottom = np.zeros_like(
        data.values,
        dtype=float,
    )

    negative_bottom = np.zeros_like(
        data.values,
        dtype=float,
    )

    for key, histogram in scaled_components.items():
        values = histogram.values

        bottom = np.where(
            values >= 0.0,
            positive_bottom,
            negative_bottom,
        )

        label = (
            labels.get(
                key,
                key,
            )
            if labels
            else key
        )

        ax.bar(
            left,
            values,
            width=widths,
            bottom=bottom,
            align="edge",
            label=label,
            linewidth=0.0,
        )

        positive_bottom += np.where(
            values > 0.0,
            values,
            0.0,
        )

        negative_bottom += np.where(
            values < 0.0,
            values,
            0.0,
        )

    # Total MC statistical uncertainty.
    if (
        mc_uncertainty_band
        and total_mc.errors is not None
    ):
        _step_band(
            ax,
            total_mc.edges,
            (
                total_mc.values
                - total_mc.errors
            ),
            (
                total_mc.values
                + total_mc.errors
            ),
            color="black",
            alpha=mc_band_alpha,
            label="MC stat. unc.",
            zorder=3,
        )

    # Data points.
    ax.errorbar(
        data.centers,
        data.values,
        xerr=0.5 * data.widths,
        yerr=data.errors,
        fmt="o",
        linestyle="none",
        markersize=4,
        color="black",
        label=data_label,
        zorder=5,
    )

    # Ratio points contain data uncertainty; MC uncertainty is represented
    # separately as a band if requested.
    ratio, ratio_error = ratio_with_uncertainty(
        data.values,
        total_mc.values,
        data.errors,
        (
            None
            if mc_uncertainty_band
            else total_mc.errors
        ),
    )

    rax.errorbar(
        data.centers,
        ratio,
        xerr=0.5 * data.widths,
        yerr=ratio_error,
        fmt="o",
        linestyle="none",
        markersize=3.5,
        color="black",
        zorder=4,
    )

    if (
        mc_uncertainty_band
        and total_mc.errors is not None
    ):
        band = _mc_ratio_band(
            total_mc.values,
            total_mc.errors,
        )

        if band is not None:
            low, high, _ = band

            _step_band(
                rax,
                total_mc.edges,
                low,
                high,
                color="black",
                alpha=mc_band_alpha,
                zorder=1,
            )

    rax.axhline(
        1.0,
        linewidth=1,
        linestyle="--",
    )

    if xlim is not None:
        ax.set_xlim(
            *xlim
        )

    if logx:
        ax.set_xscale(
            "log"
        )

    if ylim is not None:
        ax.set_ylim(
            *ylim
        )

    elif logy:
        positive_candidates = []

        data_positive = (
            data.values[
                data.values > 0.0
            ]
        )

        mc_positive = (
            positive_bottom[
                positive_bottom > 0.0
            ]
        )

        if data_positive.size:
            positive_candidates.append(
                data_positive
            )

        if mc_positive.size:
            positive_candidates.append(
                mc_positive
            )

        if positive_candidates:
            ymin = min(
                np.min(values)
                for values in positive_candidates
            )
            ymax = max(
                np.max(values)
                for values in positive_candidates
            )
        else:
            ymin = 1e-3
            ymax = 1.0

        ax.set_ylim(
            max(
                0.5 * ymin,
                1e-6,
            ),
            max(
                20.0 * ymax,
                10.0 * ymin,
            ),
        )

    else:
        data_error = (
            np.zeros_like(
                data.values
            )
            if data.errors is None
            else data.errors
        )

        ymin = min(
            0.0,
            float(
                np.nanmin(
                    negative_bottom
                )
            ),
            float(
                np.nanmin(
                    data.values
                    - data_error
                )
            ),
        )

        ymax = max(
            float(
                np.nanmax(
                    positive_bottom
                )
            ),
            float(
                np.nanmax(
                    data.values
                    + data_error
                )
            ),
            1.0,
        )

        span = max(
            ymax - ymin,
            1.0,
        )

        ax.set_ylim(
            ymin - 0.05 * span,
            ymax + 0.15 * span,
        )

    if logy:
        ax.set_yscale(
            "log"
        )

    if ratio_ylim is not None:
        rax.set_ylim(
            *ratio_ylim
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

    if title:
        ax.text(
            0.03,
            0.91,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
        )

    if legend_outside:
        handles, legend_labels = (
            ax.get_legend_handles_labels()
        )

        side_ax.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(
                0.02,
                0.98,
            ),
            borderaxespad=0.0,
            frameon=False,
            fontsize=legend_fontsize,
        )

        if side_text:
            side_ax.text(
                0.02,
                0.58,
                str(side_text),
                transform=side_ax.transAxes,
                ha="left",
                va="top",
                fontsize="small",
            )

    else:
        ax.legend(
            frameon=False,
            fontsize=legend_fontsize,
        )

    draw_cms_label(
        ax,
        cms_label=cms_label,
        lumi=lumi,
        com=com,
        data=True,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.12,
        top=0.94,
    )

    save_figure(
        fig,
        output,
    )
