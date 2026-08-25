from dataclasses import dataclass, replace
from typing import Optional
import numpy as np


@dataclass
class Hist1D:
    values: np.ndarray
    edges: np.ndarray
    errors: Optional[np.ndarray] = None
    name: str = ""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.edges = np.asarray(self.edges, dtype=float)
        if self.errors is not None:
            self.errors = np.asarray(self.errors, dtype=float)

        if len(self.edges) != len(self.values) + 1:
            raise ValueError(
                f"Hist1D '{self.name}': expected len(edges)=len(values)+1, "
                f"got {len(self.edges)} and {len(self.values)}"
            )

    @property
    def centers(self):
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self):
        return self.edges[1:] - self.edges[:-1]

    @property
    def integral(self):
        return float(np.sum(self.values))

    def scaled(self, factor):
        errors = None if self.errors is None else np.abs(factor) * self.errors
        return replace(
            self,
            values=self.values * factor,
            errors=errors,
        )

    def normalized(self):
        integral = self.integral
        if not np.isfinite(integral) or abs(integral) < 1e-15:
            return replace(self)
        return self.scaled(1.0 / integral)


@dataclass
class Hist2D:
    values: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    errors: Optional[np.ndarray] = None
    name: str = ""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.xedges = np.asarray(self.xedges, dtype=float)
        self.yedges = np.asarray(self.yedges, dtype=float)
        if self.errors is not None:
            self.errors = np.asarray(self.errors, dtype=float)

        expected = (len(self.yedges) - 1, len(self.xedges) - 1)
        if self.values.shape != expected:
            raise ValueError(
                f"Hist2D '{self.name}': expected shape {expected}, "
                f"got {self.values.shape}"
            )


@dataclass
class Hist3D:
    """
    Lightweight numpy representation of a ROOT TH3.

    Axis convention:
        values[z_bin, y_bin, x_bin]

    This extends the existing Hist2D convention:
        values[y_bin, x_bin]
    """
    values: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    zedges: np.ndarray
    errors: Optional[np.ndarray] = None
    name: str = ""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""

    def __post_init__(self):
        self.values = np.asarray(
            self.values,
            dtype=float,
        )
        self.xedges = np.asarray(
            self.xedges,
            dtype=float,
        )
        self.yedges = np.asarray(
            self.yedges,
            dtype=float,
        )
        self.zedges = np.asarray(
            self.zedges,
            dtype=float,
        )

        if self.errors is not None:
            self.errors = np.asarray(
                self.errors,
                dtype=float,
            )

        expected = (
            len(self.zedges) - 1,
            len(self.yedges) - 1,
            len(self.xedges) - 1,
        )

        if self.values.shape != expected:
            raise ValueError(
                "Hist3D '{}': expected shape {}, got {}"
                .format(
                    self.name,
                    expected,
                    self.values.shape,
                )
            )


@dataclass
class Graph1D:
    x: np.ndarray
    y: np.ndarray
    xerr_low: Optional[np.ndarray] = None
    xerr_high: Optional[np.ndarray] = None
    yerr_low: Optional[np.ndarray] = None
    yerr_high: Optional[np.ndarray] = None
    name: str = ""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

        for attr in ("xerr_low", "xerr_high", "yerr_low", "yerr_high"):
            value = getattr(self, attr)
            if value is not None:
                setattr(self, attr, np.asarray(value, dtype=float))

        if len(self.x) != len(self.y):
            raise ValueError(
                f"Graph1D '{self.name}': x/y lengths differ "
                f"({len(self.x)} vs {len(self.y)})"
            )

    @property
    def yerr(self):
        if self.yerr_low is None and self.yerr_high is None:
            return None
        low = self.yerr_low if self.yerr_low is not None else self.yerr_high
        high = self.yerr_high if self.yerr_high is not None else self.yerr_low
        return np.vstack([low, high])

    @property
    def xerr(self):
        if self.xerr_low is None and self.xerr_high is None:
            return None
        low = self.xerr_low if self.xerr_low is not None else self.xerr_high
        high = self.xerr_high if self.xerr_high is not None else self.xerr_low
        return np.vstack([low, high])


def ratio_with_uncertainty(
    numerator,
    denominator,
    numerator_error=None,
    denominator_error=None,
):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    ratio = np.full_like(numerator, np.nan, dtype=float)
    valid = np.isfinite(denominator) & (denominator != 0)
    ratio[valid] = numerator[valid] / denominator[valid]

    if numerator_error is None and denominator_error is None:
        return ratio, None

    nerr = (
        np.zeros_like(numerator)
        if numerator_error is None
        else np.asarray(numerator_error, dtype=float)
    )
    derr = (
        np.zeros_like(denominator)
        if denominator_error is None
        else np.asarray(denominator_error, dtype=float)
    )

    error = np.full_like(ratio, np.nan, dtype=float)

    # Stable absolute-error propagation:
    # r = n/d
    # sigma_r^2 = (sigma_n/d)^2 + (n sigma_d/d^2)^2
    error[valid] = np.sqrt(
        (nerr[valid] / denominator[valid]) ** 2
        + (
            numerator[valid]
            * derr[valid]
            / denominator[valid] ** 2
        ) ** 2
    )

    return ratio, error
