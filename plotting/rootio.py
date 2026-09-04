import numpy as np

from .objects import Hist1D, Hist2D, Hist3D, Graph1D


def _axis_title(axis):
    if axis is None:
        return ""
    return str(axis.GetTitle() or "")


def _read_th1(obj):
    nbins = obj.GetNbinsX()

    edges = np.array(
        [obj.GetXaxis().GetBinLowEdge(i) for i in range(1, nbins + 1)]
        + [obj.GetXaxis().GetBinUpEdge(nbins)],
        dtype=float,
    )

    values = np.array(
        [obj.GetBinContent(i) for i in range(1, nbins + 1)],
        dtype=float,
    )

    errors = np.array(
        [obj.GetBinError(i) for i in range(1, nbins + 1)],
        dtype=float,
    )

    return Hist1D(
        values=values,
        edges=edges,
        errors=errors,
        name=str(obj.GetName()),
        title=str(obj.GetTitle() or ""),
        xlabel=_axis_title(obj.GetXaxis()),
        ylabel=_axis_title(obj.GetYaxis()),
    )


def _read_th3(obj):
    """
    Convert a ROOT TH3 into Hist3D.

    Array orientation is:
        values[z, y, x]

    matching the conventions used by Hist2D and the study projection helpers.
    """
    nx = obj.GetNbinsX()
    ny = obj.GetNbinsY()
    nz = obj.GetNbinsZ()

    xedges = np.array(
        [
            obj.GetXaxis().GetBinLowEdge(i)
            for i in range(1, nx + 1)
        ]
        + [
            obj.GetXaxis().GetBinUpEdge(nx)
        ],
        dtype=float,
    )

    yedges = np.array(
        [
            obj.GetYaxis().GetBinLowEdge(i)
            for i in range(1, ny + 1)
        ]
        + [
            obj.GetYaxis().GetBinUpEdge(ny)
        ],
        dtype=float,
    )

    zedges = np.array(
        [
            obj.GetZaxis().GetBinLowEdge(i)
            for i in range(1, nz + 1)
        ]
        + [
            obj.GetZaxis().GetBinUpEdge(nz)
        ],
        dtype=float,
    )

    values = np.zeros(
        (nz, ny, nx),
        dtype=float,
    )

    errors = np.zeros(
        (nz, ny, nx),
        dtype=float,
    )

    for iz in range(1, nz + 1):
        for iy in range(1, ny + 1):
            for ix in range(1, nx + 1):
                values[
                    iz - 1,
                    iy - 1,
                    ix - 1,
                ] = obj.GetBinContent(
                    ix,
                    iy,
                    iz,
                )

                errors[
                    iz - 1,
                    iy - 1,
                    ix - 1,
                ] = obj.GetBinError(
                    ix,
                    iy,
                    iz,
                )

    return Hist3D(
        values=values,
        xedges=xedges,
        yedges=yedges,
        zedges=zedges,
        errors=errors,
        name=str(
            obj.GetName()
        ),
        title=str(
            obj.GetTitle()
            or ""
        ),
        xlabel=_axis_title(
            obj.GetXaxis()
        ),
        ylabel=_axis_title(
            obj.GetYaxis()
        ),
        zlabel=_axis_title(
            obj.GetZaxis()
        ),
    )


def _read_th2(obj):
    nx = obj.GetNbinsX()
    ny = obj.GetNbinsY()

    xedges = np.array(
        [obj.GetXaxis().GetBinLowEdge(i) for i in range(1, nx + 1)]
        + [obj.GetXaxis().GetBinUpEdge(nx)],
        dtype=float,
    )

    yedges = np.array(
        [obj.GetYaxis().GetBinLowEdge(i) for i in range(1, ny + 1)]
        + [obj.GetYaxis().GetBinUpEdge(ny)],
        dtype=float,
    )

    values = np.zeros((ny, nx), dtype=float)
    errors = np.zeros((ny, nx), dtype=float)

    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            values[iy - 1, ix - 1] = obj.GetBinContent(ix, iy)
            errors[iy - 1, ix - 1] = obj.GetBinError(ix, iy)

    return Hist2D(
        values=values,
        xedges=xedges,
        yedges=yedges,
        errors=errors,
        name=str(obj.GetName()),
        title=str(obj.GetTitle() or ""),
        xlabel=_axis_title(obj.GetXaxis()),
        ylabel=_axis_title(obj.GetYaxis()),
        zlabel=_axis_title(obj.GetZaxis()),
    )


def _read_tgraph(obj):
    n = obj.GetN()

    x = np.array([obj.GetPointX(i) for i in range(n)], dtype=float)
    y = np.array([obj.GetPointY(i) for i in range(n)], dtype=float)

    xlow = np.zeros(n, dtype=float)
    xhigh = np.zeros(n, dtype=float)
    ylow = np.zeros(n, dtype=float)
    yhigh = np.zeros(n, dtype=float)

    if obj.InheritsFrom("TGraphAsymmErrors"):
        for i in range(n):
            xlow[i] = obj.GetErrorXlow(i)
            xhigh[i] = obj.GetErrorXhigh(i)
            ylow[i] = obj.GetErrorYlow(i)
            yhigh[i] = obj.GetErrorYhigh(i)

    elif obj.InheritsFrom("TGraphErrors"):
        for i in range(n):
            ex = obj.GetErrorX(i)
            ey = obj.GetErrorY(i)

            xlow[i] = ex
            xhigh[i] = ex
            ylow[i] = ey
            yhigh[i] = ey

    else:
        xlow = None
        xhigh = None
        ylow = None
        yhigh = None

    return Graph1D(
        x=x,
        y=y,
        xerr_low=xlow,
        xerr_high=xhigh,
        yerr_low=ylow,
        yerr_high=yhigh,
        name=str(obj.GetName()),
        title=str(obj.GetTitle() or ""),
    )


def _read_tefficiency(obj):
    graph = obj.CreateGraph()
    result = _read_tgraph(graph)

    total = obj.GetTotalHistogram()
    result.xlabel = _axis_title(total.GetXaxis())
    result.ylabel = "Efficiency"

    return result


def convert_root_object(obj, object_path=""):
    """
    Convert an already-loaded ROOT object into a lightweight numpy dataclass.
    """
    if obj.InheritsFrom("TEfficiency"):
        return _read_tefficiency(obj)

    # TH3 before TH2 before TH1 because of ROOT inheritance.
    if obj.InheritsFrom("TH3"):
        return _read_th3(obj)

    if obj.InheritsFrom("TH2"):
        return _read_th2(obj)

    if obj.InheritsFrom("TH1"):
        return _read_th1(obj)

    if obj.InheritsFrom("TGraph"):
        return _read_tgraph(obj)

    label = object_path or str(obj.GetName())
    raise TypeError(
        "Unsupported ROOT object '{}' of class {}".format(
            label,
            obj.ClassName(),
        )
    )


class RootFileReader(object):
    """
    Persistent ROOT-file reader.

    The ROOT file is opened once and stays open until close() or context-manager
    exit. Converted objects are cached in memory, so repeated requests for the
    same object do not call TFile::Get again.

    This is especially important for EOS/XRootD files.
    """

    def __init__(self, file_path, cache=True):
        self.file_path = str(file_path)
        self.cache_enabled = bool(cache)
        self._file = None
        self._cache = {}

    def open(self):
        if self._file is not None:
            return self

        import ROOT

        self._file = ROOT.TFile.Open(
            self.file_path,
            "READ",
        )

        if not self._file or self._file.IsZombie():
            self._file = None
            raise OSError(
                "Could not open ROOT file: {}".format(
                    self.file_path
                )
            )

        return self

    def close(self):
        if self._file is not None:
            self._file.Close()
            self._file = None

    def clear_cache(self):
        self._cache.clear()

    def get(self, object_path):
        object_path = str(object_path)

        if self.cache_enabled and object_path in self._cache:
            return self._cache[object_path]

        if self._file is None:
            self.open()

        obj = self._file.Get(object_path)

        if not obj:
            raise KeyError(
                "Object '{}' not found in '{}'".format(
                    object_path,
                    self.file_path,
                )
            )

        converted = convert_root_object(
            obj,
            object_path=object_path,
        )

        if self.cache_enabled:
            self._cache[object_path] = converted

        return converted

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def read_root_object(file_path, object_path):
    """
    Convenience one-shot reader.

    For one or two objects this is convenient. For many objects from the same
    file, use RootFileReader instead so the ROOT file is opened only once.
    """
    with RootFileReader(file_path, cache=False) as reader:
        return reader.get(object_path)
