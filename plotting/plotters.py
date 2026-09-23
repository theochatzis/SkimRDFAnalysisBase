from pathlib import Path

import numpy as np

from .runtime import configure_matplotlib_runtime

configure_matplotlib_runtime()

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep

from .objects import Hist1D, ratio_with_uncertainty


_STYLE_INITIALIZED = False


# ----------------------------------------------------------------------
# Shared style defaults
#
# Every plotting entry point in this package applies these, and each one can
# be overridden per call. They exist because the mplhep CMS style is sized
# for a single full-page figure, which is not what a few hundred small PNGs
# want.
# ----------------------------------------------------------------------

# "CMS Preliminary ... (13.6 TeV)" header. The mplhep default overruns the
# axes width once the experiment label carries a suffix such as "Simulation".
CMS_LABEL_FONTSIZE = 16

# Tick labels switch to scientific notation once they would need more than
# this many integer digits. Without it a five-digit count such as 15000 makes
# the tick labels wide enough to push the axis label off the canvas.
SCI_DIGITS = 3

# Where the "x 10^n" factor of a scientific y axis is drawn, in axes
# coordinates. Just outside the left spine and just above the top one: the
# CMS label starts at x = 0, so a right-aligned string at a slightly negative
# x lands in the margin above the tick labels without touching it.
SCI_EXPONENT_POSITION = (-0.01, 1.01)

# Decade subdivisions that get a label on a log axis, so the eye has
# something between the powers of ten: 20, 50, 200, 500, ... Adding 3 as well
# is tempting but the labels start touching in the compressed decades of a
# wide axis, so two per decade is the readable default.
LOG_MINOR_SUBS = (2.0, 5.0)

# Subdivisions that get a tick *mark*. Every one of them, labelled or not:
# the marks are what makes a log axis readable between the decades, and
# leaving out 3, 4, 6, 7, 8, 9 turns the axis into a handful of unrelated
# stops. Only LOG_MINOR_SUBS of these are given a label.
LOG_MINOR_TICK_SUBS = (
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
)

# Extra figure width and side-column share used when the legend is moved out
# of the axes. The defaults have to fit a label such as
# "$150 < p_{T}^{Z} < 300$ GeV" without clipping.
# 0.52 of a 12.4 in figure leaves the main axes at 8.2 in, exactly the width
# it has when the legend sits inside, so the two layouts stay comparable.
SIDE_PANEL_WIDTH = 0.52
SIDE_PANEL_FIGSIZE = (12.4, 7.2)

# With a side panel the legend is meant to reach the figure edge, so the
# right margin is nearly zero.
SIDE_PANEL_MARGINS = {"right": 0.995}

# Plain numbers stay readable up to this many decades; beyond it the major
# ticks fall back to 10^n.
LOG_PLAIN_DECADES = 4

# Filled legend swatches - stack components, uncertainty bands - are drawn
# without an edge, which leaves a pale fill with no boundary against the
# white background. Outline them.
LEGEND_PATCH_EDGECOLOR = "black"
LEGEND_PATCH_LINEWIDTH = 0.8

# Figure margins. The left margin has to clear the y-axis label *and* its
# tick labels at CMS font sizes.
FIGURE_MARGINS = {
    "left": 0.16,
    "right": 0.97,
    "bottom": 0.13,
    "top": 0.93,
}


def use_hep_style():
    global _STYLE_INITIALIZED

    if _STYLE_INITIALIZED:
        return

    hep.style.use("CMS")
    _STYLE_INITIALIZED = True


def style_legend_patches(
    legend,
    *,
    edgecolor=LEGEND_PATCH_EDGECOLOR,
    linewidth=LEGEND_PATCH_LINEWIDTH,
    enabled=True,
):
    """
    Outline the filled swatches in a legend.

    Stack components and uncertainty bands are drawn without an edge, so in
    the legend a pale fill has no boundary against the white background.
    Only Patch handles are touched, which leaves errorbar and line handles
    (data points, reference lines) exactly as they were.
    """
    if legend is None or not enabled or not edgecolor:
        return legend

    # Renamed in matplotlib 3.7; keep working on either.
    handles = getattr(
        legend,
        "legend_handles",
        None,
    )

    if handles is None:
        handles = getattr(
            legend,
            "legendHandles",
            (),
        )

    for handle in handles:
        if isinstance(handle, mpatches.Patch):
            handle.set_edgecolor(edgecolor)
            handle.set_linewidth(linewidth)

    return legend


def apply_figure_margins(fig, margins=None):
    """Apply the shared margins, optionally overriding individual sides."""
    values = dict(FIGURE_MARGINS)

    if margins:
        values.update(margins)

    fig.subplots_adjust(**values)


def _plain_tick(value, _position=None):
    """Tick label without exponent notation: 20, 50, 200, 2000."""
    return "{:g}".format(value)


def _axis_of(ax, which):
    return ax.xaxis if which == "x" else ax.yaxis


def _major_label_size(ax, which):
    """
    Point size the major tick labels of this axis are actually drawn at.

    Read from a tick rather than from rcParams so that an axis whose major
    labels were resized by the caller still reports the size in effect.
    """
    axis = _axis_of(ax, which)

    for tick in axis.get_major_ticks():
        return tick.label1.get_fontsize()

    return plt.rcParams[
        "{}tick.labelsize".format(which)
    ]


def _log_minor_tick(subs):
    """
    Formatter that labels only the listed subdivisions of each decade.

    The minor locator puts a mark on every subdivision; this decides which of
    those marks also carry a number. It works from the mantissa rather than
    from the value, so a single formatter stays valid across every decade of
    the axis: 0.2, 2, 20 and 200 are labelled, 3 and 300 never are.
    """

    def formatter(value, _position=None):
        if value <= 0.0:
            return ""

        mantissa = value / 10.0 ** np.floor(
            np.log10(value)
        )

        for sub in subs:
            if abs(mantissa - sub) < 1e-6 * sub:
                return _plain_tick(value)

        return ""

    return formatter


def _matching_minor_pad(ax, which):
    """
    Base pad that puts the minor tick labels as far from the spine as the
    major ones.

    A tick label sits at `get_pad() + get_tick_padding()` from the spine, and
    the CMS style gives the two sets of ticks different pads - 6.0 against 3.4
    on x - so left alone the subdivision labels creep towards the axis while
    the decade labels stay put. Returns None when the axis has no ticks to
    measure yet.
    """
    axis = _axis_of(ax, which)

    major = axis.get_major_ticks()
    minor = axis.get_minor_ticks()

    if not major or not minor:
        return None

    return (
        major[0].get_pad()
        + major[0].get_tick_padding()
        - minor[0].get_tick_padding()
    )


def _scale_of(ax, which):
    return (
        ax.get_xscale()
        if which == "x"
        else ax.get_yscale()
    )


def _plain_digits(value):
    """Digit characters in the plain rendering of a tick value."""
    return sum(
        character.isdigit()
        for character in "{:g}".format(value)
    )


def _widest_plain_tick(axis, low, high):
    """
    Digits in the widest plain tick label the axis would draw.

    Only the ticks inside the view interval count: the locator happily
    returns ticks beyond both ends, and a label that is never drawn should
    not decide the notation for the ones that are.
    """
    lower, upper = sorted((low, high))

    widest = 0

    for value in axis.get_majorticklocs():
        if not (lower <= value <= upper):
            continue
        widest = max(
            widest,
            _plain_digits(value),
        )

    return widest


def apply_scientific_ticks(
    ax,
    which="y",
    *,
    digits=SCI_DIGITS,
    enabled=True,
):
    """
    Use scientific notation once a tick label would need more than `digits`
    digits, so that neither a large count nor a small fraction widens the
    margin indefinitely.

    Silently does nothing on a log axis, where the formatter does not apply.
    """
    if not enabled or digits is None:
        return

    if _scale_of(ax, which) != "linear":
        return

    low, high = (
        ax.get_xlim()
        if which == "x"
        else ax.get_ylim()
    )

    largest = max(
        abs(low),
        abs(high),
    )

    if not np.isfinite(largest) or largest <= 0.0:
        return

    order = int(
        np.floor(
            np.log10(largest)
        )
    )

    axis = _axis_of(ax, which)

    # Two independent reasons to switch notation.
    #
    # The order of magnitude bounds the integer digits: with digits=3, 999
    # stays as it is and 1000 becomes 1.0 x 10^3. It also catches a range so
    # small that "{:g}" would itself fall back to exponent form, where
    # counting digits in the rendering no longer measures the width.
    too_many_orders = not (-digits <= order <= digits - 1)

    # Below 1 the order says nothing about the width, because the digits are
    # decimal places rather than integer ones: an axis topping out at 0.002
    # has order -3 yet labels its ticks 0.00025, which is six digits. So ask
    # the ticks themselves how wide they would print.
    too_many_digits = (
        _widest_plain_tick(axis, low, high) > digits
    )

    if not (too_many_orders or too_many_digits):
        return

    factor = 10.0 ** order

    axis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _position: "{:g}".format(
                value / factor
            )
        )
    )

    exponent = r"$\times 10^{{{}}}$".format(order)

    if which == "y":
        # Above the top of the y axis, right-aligned just outside the frame,
        # so it sits over the tick-label column and clears the CMS label that
        # starts at the left spine. Not matplotlib's own offset text, which is
        # drawn inside that corner and collides with it.
        ax.text(
            SCI_EXPONENT_POSITION[0],
            SCI_EXPONENT_POSITION[1],
            exponent,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=_major_label_size(ax, which),
        )
        return

    # On x there is no corner to put it in, so it stays in the axis label.
    label = axis.get_label().get_text()

    axis.set_label_text(
        "{} {}".format(label, exponent).strip()
    )


def apply_log_minor_ticks(
    ax,
    which="x",
    *,
    subs=LOG_MINOR_SUBS,
    tick_subs=LOG_MINOR_TICK_SUBS,
    enabled=True,
    labelsize=None,
):
    """
    Draw every subdivision inside each decade of a log axis and label a few.

    Over a narrow range the major ticks become plain numbers too, because
    "100, 1000" reads better than "10^2, 10^3"; over a wide range the powers
    of ten are kept.

    Every sub in `tick_subs` gets a tick mark; only those in `subs` get a
    number under it.

    The subdivision labels are ordinary axis labels, not annotations, so by
    default they are drawn at the same size as the decade labels, and at the
    same distance from the spine: an axis reading "2 5 10 20 50 100" should
    not change type size or drift towards the frame halfway along. Pass
    `labelsize` explicitly to override the size.
    """
    if not enabled or not tick_subs:
        return

    if _scale_of(ax, which) != "log":
        return

    axis = _axis_of(ax, which)

    low, high = (
        ax.get_xlim()
        if which == "x"
        else ax.get_ylim()
    )

    if not (low > 0.0 and high > low):
        return

    decades = np.log10(high / low)

    axis.set_major_locator(
        mticker.LogLocator(base=10.0),
    )

    if decades <= LOG_PLAIN_DECADES:
        axis.set_major_formatter(
            mticker.FuncFormatter(_plain_tick),
        )
    else:
        axis.set_major_formatter(
            mticker.LogFormatterSciNotation(base=10.0),
        )

    # Mark every subdivision, label only the ones asked for. The two are
    # separate on purpose: a log axis that jumps 10, 20, 50, 100 with nothing
    # drawn in between reads as a linear axis with odd spacing.
    axis.set_minor_locator(
        mticker.LogLocator(
            base=10.0,
            subs=tick_subs,
            numticks=100,
        ),
    )

    axis.set_minor_formatter(
        mticker.FuncFormatter(
            _log_minor_tick(subs),
        ),
    )

    # Resolve after the major formatter and locator are in place, so the size
    # comes from the ticks this axis will actually draw.
    if labelsize is None:
        labelsize = _major_label_size(ax, which)

    params = {"labelsize": labelsize}

    pad = _matching_minor_pad(ax, which)

    if pad is not None:
        params["pad"] = pad

    ax.tick_params(
        axis=which,
        which="minor",
        **params,
    )


def apply_axis_style(
    ax,
    *,
    sci_digits=SCI_DIGITS,
    log_minor_labels=True,
    sci_axis="y",
    log_axis="x",
):
    """
    The shared tick styling, applied after the scales and limits are set.

    Call this on whichever axes actually shows the labels: for a main+ratio
    layout that is the ratio panel for x and the main panel for y.
    """
    apply_scientific_ticks(
        ax,
        sci_axis,
        digits=sci_digits,
    )

    apply_log_minor_ticks(
        ax,
        log_axis,
        enabled=log_minor_labels,
    )


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
    fontsize=CMS_LABEL_FONTSIZE,
):
    hep.cms.label(
        cms_label,
        data=data,
        lumi=lumi,
        com=com,
        ax=ax,
        fontsize=fontsize,
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
    sci_digits=SCI_DIGITS,
    log_minor_labels=True,
    cms_label_fontsize=CMS_LABEL_FONTSIZE,
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
            (SIDE_PANEL_FIGSIZE[0], 7.6)
            if legend_outside
            else (8.2, 7.6)
        ),
        side_panel=legend_outside,
        side_width=SIDE_PANEL_WIDTH,
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

        style_legend_patches(
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
        style_legend_patches(
            ax.legend(
                ncol=2,
                fontsize=legend_fontsize,
                frameon=False,
            )
        )

    if ratio_legend:
        style_legend_patches(
            rax.legend(
                ncol=max(
                    1,
                    len(keys),
                ),
                fontsize=legend_fontsize,
                frameon=False,
            )
        )

    apply_axis_style(
        ax,
        sci_digits=sci_digits,
        log_minor_labels=False,
    )
    apply_axis_style(
        rax,
        sci_digits=None,
        log_minor_labels=log_minor_labels,
    )

    draw_cms_label(
        ax,
        cms_label=cms_label,
        lumi=lumi,
        com=com,
        data=True,
        fontsize=cms_label_fontsize,
    )

    apply_figure_margins(
        fig,
        SIDE_PANEL_MARGINS if legend_outside else None,
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
    sci_digits=SCI_DIGITS,
    log_minor_labels=True,
    cms_label_fontsize=CMS_LABEL_FONTSIZE,
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
        sci_digits=sci_digits,
        log_minor_labels=log_minor_labels,
        cms_label_fontsize=cms_label_fontsize,
    )
