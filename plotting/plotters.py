from pathlib import Path

import numpy as np

from .runtime import configure_matplotlib_runtime

configure_matplotlib_runtime()

import matplotlib.pyplot as plt
import mplhep as hep

from .objects import Hist1D, ratio_with_uncertainty


_STYLE_INITIALIZED = False


def use_hep_style():
    global _STYLE_INITIALIZED

    if _STYLE_INITIALIZED:
        return

    hep.style.use("CMS")
    _STYLE_INITIALIZED = True


def save_figure(fig, output, dpi=150):
    if isinstance(output, (str, Path)):
        outputs = [output]
    else:
        outputs = list(output)

    for destination in outputs:
        destination = Path(destination)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        kwargs = {}

        if destination.suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
        ):
            kwargs["dpi"] = dpi

        # Avoid bbox_inches="tight": it causes an extra render pass.
        fig.savefig(
            destination,
            **kwargs
        )

    plt.close(fig)


def draw_cms_label(
    ax,
    *,
    cms_label="Preliminary",
    lumi=None,
    com=13.6,
    data=True,
):
    hep.cms.label(
        cms_label,
        data=data,
        lumi=lumi,
        com=com,
        ax=ax,
    )


def ratio_axes(
    figsize=(10.0, 7.6),
    ratio_height=0.28,
    side_panel=False,
    side_width=0.34,
):
    """
    Create the standard main + ratio layout.

    If side_panel=True, reserve a dedicated right-hand column for legends,
    selection tags, and other annotations. Keeping the legend inside a real
    axes avoids clipping when saving without bbox_inches="tight".
    """
    fig = plt.figure(
        figsize=figsize
    )

    if side_panel:
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[
                1.0 - ratio_height,
                ratio_height,
            ],
            width_ratios=[
                1.0,
                side_width,
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

    else:
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[
                1.0 - ratio_height,
                ratio_height,
            ],
            hspace=0.04,
        )

        ax = fig.add_subplot(
            gs[0]
        )

        rax = fig.add_subplot(
            gs[1],
            sharex=ax,
        )

        side_ax = None

    plt.setp(
        ax.get_xticklabels(),
        visible=False,
    )

    return (
        fig,
        ax,
        rax,
        side_ax,
    )

def finite_extent(
    values,
    errors=None,
    *,
    mask=None,
    include_errors=True,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if mask is None:
        mask = np.isfinite(
            values
        )
    else:
        mask = (
            np.asarray(mask, dtype=bool)
            & np.isfinite(values)
        )

    if errors is None:
        errors = np.zeros_like(
            values
        )
    else:
        errors = np.asarray(
            errors,
            dtype=float,
        )

        mask = (
            mask
            & np.isfinite(errors)
        )

    if not np.any(mask):
        return None

    selected = values[mask]

    if include_errors:
        selected_error = np.abs(
            errors[mask]
        )

        return (
            float(
                np.min(
                    selected
                    - selected_error
                )
            ),
            float(
                np.max(
                    selected
                    + selected_error
                )
            ),
        )

    return (
        float(np.min(selected)),
        float(np.max(selected)),
    )

## Automatic range tools
def auto_range(
    categories,
    *,
    include_errors=True,
    padding=0.10,
    max_rel_uncertainty=None,
    zero_threshold=1e-12,
    log=False,
):
    """
    Automatically determine a common y-axis range for multiple datasets.

    Each entry in `categories` should be:
        (values, errors, mask)
    
    note that so for one dataset only you could do just
    auto_range([(values,errors,None)])

    where:
        values : array of central values
        errors : array of uncertainties, or None
        mask   : boolean array selecting valid plotted points, or None

    The range is determined from all statistically useful points using:
        lower edge = value - error
        upper edge = value + error

    Points do not contribute to the automatic range if:
        - they are masked out;
        - their value is non-finite;
        - their value is effectively zero;
        - their uncertainty is non-finite;
        - their relative uncertainty exceeds `max_rel_uncertainty`.

    For logarithmic axes:
        - only points with (value - error) > 0 are allowed to define
          the lower range;
        - padding is applied multiplicatively in log10 space.

    Importantly, excluding a point from auto-ranging does NOT remove it
    from the plot. It only prevents that point from controlling the axis.
    """

    all_low = []
    all_high = []

    # --------------------------------------------------------
    # Loop over all Data / MC / method categories.
    # The final range will be common to all of them.
    # --------------------------------------------------------
    for values, errors, mask in categories:

        values = np.asarray(
            values,
            dtype=float,
        )

        # ----------------------------------------------------
        # Start from the supplied plotting mask.
        # If no mask is provided, initially accept every point.
        # ----------------------------------------------------
        if mask is None:
            valid = np.ones(
                values.shape,
                dtype=bool,
            )
        else:
            valid = np.asarray(
                mask,
                dtype=bool,
            ).copy()

        # ----------------------------------------------------
        # Remove non-finite and effectively empty points.
        # A histogram/profile bin with exactly zero content
        # should not force the automatic range toward zero.
        # ----------------------------------------------------
        valid &= np.isfinite(
            values
        )

        valid &= (
            np.abs(values)
            > zero_threshold
        )

        # ----------------------------------------------------
        # Include uncertainties in the visible range when they
        # are available:
        #
        #     ymin <- value - error
        #     ymax <- value + error
        # ----------------------------------------------------
        if (
            include_errors
            and errors is not None
        ):
            errors = np.asarray(
                errors,
                dtype=float,
            )

            valid &= np.isfinite(
                errors
            )

            valid &= (
                errors >= 0.0
            )

            # ------------------------------------------------
            # Prevent statistically poorly measured outliers
            # from determining the plot range.
            #
            # Example with max_rel_uncertainty = 0.3:
            #
            #     1.0 +/- 0.1  -> kept
            #     1.0 +/- 1.0  -> ignored for auto-ranging
            # ------------------------------------------------
            if max_rel_uncertainty is not None:

                rel_uncertainty = np.full(
                    values.shape,
                    np.inf,
                    dtype=float,
                )

                nonzero = (
                    np.abs(values)
                    > zero_threshold
                )

                rel_uncertainty[nonzero] = (
                    errors[nonzero]
                    / np.abs(
                        values[nonzero]
                    )
                )

                valid &= (
                    rel_uncertainty
                    <= max_rel_uncertainty
                )

            low = (
                values
                - errors
            )

            high = (
                values
                + errors
            )

        else:
            low = values.copy()
            high = values.copy()

        # ----------------------------------------------------
        # A logarithmic y-axis cannot use zero or negative
        # values.
        #
        # Do NOT replace such values by something tiny like
        # 1e-12, because that would artificially produce ranges
        # such as 1e-12 -> 1e4.
        #
        # Instead, simply prevent those points from determining
        # the automatic logarithmic range.
        # ----------------------------------------------------
        if log:
            valid &= (
                low
                > zero_threshold
            )

            valid &= (
                high
                > zero_threshold
            )

        # ----------------------------------------------------
        # This particular category may contain no useful points.
        # In that case simply skip it.
        # ----------------------------------------------------
        if not np.any(
            valid
        ):
            continue

        all_low.append(
            low[valid]
        )

        all_high.append(
            high[valid]
        )

    # --------------------------------------------------------
    # None of the categories contained a useful point.
    # Let the caller decide what fallback range to use.
    # --------------------------------------------------------
    if not all_low:
        return None

    # --------------------------------------------------------
    # Find the global extrema across ALL categories.
    # --------------------------------------------------------
    ymin = min(
        np.min(values)
        for values in all_low
    )

    ymax = max(
        np.max(values)
        for values in all_high
    )

    # --------------------------------------------------------
    # Add some visual padding.
    #
    # For log plots, padding must be multiplicative rather than
    # additive. We therefore work in log10 space.
    # --------------------------------------------------------
    if log:

        log_min = np.log10(
            ymin
        )

        log_max = np.log10(
            ymax
        )

        log_span = (
            log_max
            - log_min
        )

        # Protect the special case where all useful points have
        # approximately the same value.
        if (
            not np.isfinite(log_span)
            or log_span <= 0.0
        ):
            log_span = 1.0

        ymin = 10.0 ** (
            log_min
            - padding * log_span
        )

        ymax = 10.0 ** (
            log_max
            + padding * log_span
        )

    else:

        span = (
            ymax
            - ymin
        )

        # Protect the special case where all useful points have
        # approximately the same value.
        if (
            not np.isfinite(span)
            or span <= 0.0
        ):
            span = max(
                abs(ymin),
                abs(ymax),
                1.0,
            )

        ymin -= (
            padding
            * span
        )

        ymax += (
            padding
            * span
        )

    return (
        ymin,
        ymax,
    )

def hist_visible_mask(
    hist,
    xlim=None,
    *,
    ignore_empty=True,
):
    mask = np.isfinite(
        hist.values
    )

    if hist.errors is not None:
        mask = (
            mask
            & np.isfinite(
                hist.errors
            )
        )

    if ignore_empty:
        errors = (
            np.zeros_like(
                hist.values
            )
            if hist.errors is None
            else hist.errors
        )

        mask = (
            mask
            & (
                (hist.values != 0)
                | (errors != 0)
            )
        )

    if xlim is not None:
        mask = (
            mask
            & (
                hist.centers
                >= xlim[0]
            )
            & (
                hist.centers
                <= xlim[1]
            )
        )

    return mask


def _step_band(
    ax,
    edges,
    low,
    high,
    *,
    color=None,
    alpha=0.20,
    label=None,
    zorder=1,
):
    """
    Draw a histogram-shaped uncertainty band.

    low/high have one value per bin; edges has N+1 entries.
    """
    edges = np.asarray(
        edges,
        dtype=float,
    )
    low = np.asarray(
        low,
        dtype=float,
    )
    high = np.asarray(
        high,
        dtype=float,
    )

    if len(edges) != len(low) + 1:
        raise ValueError(
            "Band edges must have len(values)+1 entries."
        )

    if len(low) == 0:
        return None

    low_step = np.r_[
        low,
        low[-1],
    ]
    high_step = np.r_[
        high,
        high[-1],
    ]

    return ax.fill_between(
        edges,
        low_step,
        high_step,
        step="post",
        color=color,
        alpha=alpha,
        linewidth=0,
        label=label,
        zorder=zorder,
    )


def _mc_ratio_band(
    mc_values,
    mc_errors,
):
    """
    Return the relative MC uncertainty band around one.

        low  = 1 - sigma_MC / |MC|
        high = 1 + sigma_MC / |MC|

    Invalid/empty bins become NaN and are not drawn.
    """
    mc_values = np.asarray(
        mc_values,
        dtype=float,
    )

    if mc_errors is None:
        return None

    mc_errors = np.asarray(
        mc_errors,
        dtype=float,
    )

    relative = np.full_like(
        mc_values,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(mc_values)
        & np.isfinite(mc_errors)
        & (mc_values != 0)
    )

    relative[valid] = (
        np.abs(mc_errors[valid])
        / np.abs(mc_values[valid])
    )

    return (
        1.0 - relative,
        1.0 + relative,
        valid,
    )


def plot_hist1d_methods_data_mc(
    data_methods,
    mc_methods,
    output,
    *,
    labels=None,
    xlabel="",
    ylabel="Events",
    ratio_label="Data / MC",
    normalize_mc_to_data=False,
    xlim=None,
    ylim=None,
    auto_y=True,
    y_padding=0.12,
    ratio_ylim=None,
    auto_ratio_y=True,
    ratio_padding=0.12,
    mc_ratio_uncertainty_band=False,
    mc_ratio_band_alpha=0.20,
    logx=False,
    logy=False,
    miny_log_tolerance=1e-6, #small number that shows that if logy is activated and lower y-lim < miny_log_tolerance change to miny_log_tolerance
    legend_outside=False,
    legend_fontsize="x-small",
    ratio_legend=False,
    side_text=None,
    cms_label="Preliminary",
    lumi=None,
    com=13.6,
    title=None,
):
    """
    Compare multiple methods.

    Main panel:
      Data -> points with error bars
      MC   -> histogram-style step curves (no point-to-point interpolation)

    Ratio panel:
      Data/MC -> points, not connected lines

    If mc_ratio_uncertainty_band=True:
      * ratio-point errors contain the Data uncertainty only;
      * a shaded band around 1 shows the MC uncertainty separately.

    If False:
      * ratio-point errors contain the usual propagated Data+MC uncertainty.
    """
    use_hep_style()

    keys = list(
        data_methods.keys()
    )

    if keys != list(
        mc_methods.keys()
    ):
        raise ValueError(
            "Data and MC method ordering differs."
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

    colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()
        .get(
            "color",
            [],
        )
    )

    main_categories = []
    ratio_categories = []

    for index, key in enumerate(
        keys
    ):
        data = data_methods[key]
        mc = mc_methods[key]

        if not np.allclose(
            data.edges,
            mc.edges,
        ):
            raise ValueError(
                "Binning differs for method '{}'.".format(
                    key
                )
            )

        if normalize_mc_to_data:
            dsum = np.sum(
                data.values
            )
            msum = np.sum(
                mc.values
            )

            if (
                np.isfinite(dsum)
                and np.isfinite(msum)
                and msum != 0
            ):
                scale = (
                    dsum / msum
                )

                mc_values = (
                    mc.values
                    * scale
                )

                mc_errors = (
                    None
                    if mc.errors is None
                    else np.abs(scale)
                    * mc.errors
                )

            else:
                mc_values = (
                    mc.values
                )
                mc_errors = (
                    mc.errors
                )
        else:
            mc_values = (
                mc.values
            )
            mc_errors = (
                mc.errors
            )

        color = (
            colors[
                index % len(colors)
            ]
            if colors
            else None
        )

        label = (
            labels.get(
                key,
                key,
            )
            if labels
            else key
        )

        # MC is a histogram, not a line connecting bin centers.
        hep.histplot(
            (
                mc_values,
                mc.edges,
            ),
            histtype="step",
            linewidth=2,
            color=color,
            label="MC {}".format(
                label
            ),
            ax=ax,
        )

        # Data remains points.
        ax.errorbar(
            data.centers,
            data.values,
            yerr=data.errors,
            fmt="o",
            linestyle="none",
            markersize=4,
            color=color,
            label="Data {}".format(
                label
            ),
            zorder=4,
        )

        # Standard ratio treatment:
        # when an MC band is requested, keep denominator uncertainty out
        # of the point error bars and show it separately as a band.
        denominator_error_for_points = (
            None
            if mc_ratio_uncertainty_band
            else mc_errors
        )

        ratio, ratio_error = (
            ratio_with_uncertainty(
                data.values,
                mc_values,
                data.errors,
                denominator_error_for_points,
            )
        )

        rax.errorbar(
            data.centers,
            ratio,
            yerr=ratio_error,
            fmt="o",
            linestyle="none",
            markersize=3.5,
            color=color,
            label=label,
            zorder=4,
        )

        if mc_ratio_uncertainty_band:
            band = _mc_ratio_band(
                mc_values,
                mc_errors,
            )

            if band is not None:
                band_low, band_high, band_valid = band

                _step_band(
                    rax,
                    mc.edges,
                    band_low,
                    band_high,
                    color=color,
                    alpha=mc_ratio_band_alpha,
                    zorder=1,
                )

                ratio_categories.append(
                    (
                        np.ones_like(
                            mc_values
                        ),
                        np.where(
                            band_valid,
                            0.5
                            * (
                                band_high
                                - band_low
                            ),
                            np.nan,
                        ),
                        band_valid,
                    )
                )

        data_mask = hist_visible_mask(
            data,
            xlim=xlim,
        )

        mc_mask = (
            np.isfinite(mc_values)
        )

        if mc_errors is not None:
            mc_mask = (
                mc_mask
                & np.isfinite(
                    mc_errors
                )
            )

        mc_nonzero = (
            mc_values != 0
        )

        if mc_errors is None:
            mc_error_nonzero = np.zeros_like(
                mc_values,
                dtype=bool,
            )
        else:
            mc_error_nonzero = (
                mc_errors != 0
            )

        mc_mask = (
            mc_mask
            & (
                mc_nonzero
                | mc_error_nonzero
            )
        )

        if xlim is not None:
            mc_mask = (
                mc_mask
                & (
                    mc.centers
                    >= xlim[0]
                )
                & (
                    mc.centers
                    <= xlim[1]
                )
            )

        main_categories.append(
            (
                data.values,
                data.errors,
                data_mask,
            )
        )

        main_categories.append(
            (
                mc_values,
                mc_errors,
                mc_mask,
            )
        )

        ratio_mask = (
            np.isfinite(ratio)
        )

        if ratio_error is not None:
            ratio_mask = (
                ratio_mask
                & np.isfinite(
                    ratio_error
                )
            )

        if xlim is not None:
            ratio_mask = (
                ratio_mask
                & (
                    data.centers
                    >= xlim[0]
                )
                & (
                    data.centers
                    <= xlim[1]
                )
            )

        ratio_categories.append(
            (
                ratio,
                ratio_error,
                ratio_mask,
            )
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
    if logy:
        # In this case check if the lowest y-lim is close to zero
        ymin_, ymax_ = ax.get_ylim()

        if ymin_ < miny_log_tolerance:
            ax.set_ylim(
                miny_log_tolerance,
                ymax_,
        )
        y_padding=2.0*y_padding
    
    if auto_y:
        ax.set_ylim(
            *auto_range(
                main_categories,
                include_errors=True,
                padding=y_padding,
                max_rel_uncertainty=0.5,
                log=logy
            )
        )
    
    if logy:
        ax.set_yscale(
            "log"
        )
    
    if ratio_ylim is not None:
        rax.set_ylim(
            *ratio_ylim
        )
    elif auto_ratio_y:
        print(auto_range(
                ratio_categories,
                include_errors=True,
                padding=ratio_padding,
                reference=1.0,
                minimum_span=0.02,
            ))
        rax.set_ylim(
            *auto_range(
                ratio_categories,
                include_errors=True,
                padding=ratio_padding,
                reference=1.0,
                minimum_span=0.02,
            )
        )

    rax.axhline(
        1.0,
        linewidth=1,
        linestyle="--",
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
            ncol=1,
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
            ncol=2,
            fontsize=legend_fontsize,
            frameon=False,
        )

    if ratio_legend:
        rax.legend(
            ncol=max(
                1,
                len(keys),
            ),
            fontsize=legend_fontsize,
            frameon=False,
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


def plot_hist1d_data_mc(
    data,
    mc,
    output,
    *,
    xlabel="",
    ylabel="Events",
    ratio_label="Data / MC",
    normalize_mc_to_data=False,
    xlim=None,
    ylim=None,
    ratio_ylim=(0.5, 1.5),
    mc_ratio_uncertainty_band=False,
    logx=False,
    legend_outside=False,
    legend_fontsize="x-small",
    ratio_legend=False,
    side_text=None,
    cms_label="Preliminary",
    lumi=None,
    com=13.6,
    title=None,
):
    from collections import OrderedDict

    plot_hist1d_methods_data_mc(
        OrderedDict([
            ("value", data),
        ]),
        OrderedDict([
            ("value", mc),
        ]),
        output,
        labels={
            "value": "",
        },
        xlabel=xlabel,
        ylabel=ylabel,
        ratio_label=ratio_label,
        normalize_mc_to_data=normalize_mc_to_data,
        xlim=xlim,
        ylim=ylim,
        auto_y=(
            ylim is None
        ),
        ratio_ylim=ratio_ylim,
        auto_ratio_y=(
            ratio_ylim is None
        ),
        mc_ratio_uncertainty_band=mc_ratio_uncertainty_band,
        logx=logx,
        legend_outside=legend_outside,
        legend_fontsize=legend_fontsize,
        ratio_legend=ratio_legend,
        side_text=side_text,
        cms_label=cms_label,
        lumi=lumi,
        com=com,
        title=title,
    )
