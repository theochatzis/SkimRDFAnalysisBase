"""
Matplotlib runtime configuration for SkimRDFAnalysisBase.

The important lxplus behavior is:

  1. redirect matplotlib/XDG cache directories to local temporary storage;
  2. explicitly call matplotlib.use("Agg", force=True);
  3. do this before pyplot or mplhep is imported.

This avoids the O(1 minute) first-figure delay that can occur when font/config
caches live on EOS/AFS.
"""

import os


_CONFIGURED = False


def configure_matplotlib_runtime():
    global _CONFIGURED

    if _CONFIGURED:
        return

    enabled = os.environ.get(
        "SKIMRDF_MPL_LOCAL_CACHE",
        "1",
    ).strip().lower()

    if enabled not in ("0", "false", "no", "off"):
        user = os.environ.get("USER", "user")
        tmp_root = os.environ.get("TMPDIR", "/tmp")

        cache_root = os.path.join(
            tmp_root,
            user,
            "skimrdf_matplotlib",
        )

        mplconfig = os.environ.get(
            "SKIMRDF_MPLCONFIGDIR",
            os.path.join(cache_root, "mplconfig"),
        )

        xdg_cache = os.path.join(
            cache_root,
            "xdg_cache",
        )

        xdg_config = os.path.join(
            cache_root,
            "xdg_config",
        )

        for directory in (
            mplconfig,
            xdg_cache,
            xdg_config,
        ):
            os.makedirs(
                directory,
                exist_ok=True,
            )

        os.environ["MPLCONFIGDIR"] = mplconfig
        os.environ["XDG_CACHE_HOME"] = xdg_cache
        os.environ["XDG_CONFIG_HOME"] = xdg_config

    os.environ["MPLBACKEND"] = "Agg"

    import matplotlib

    matplotlib.use(
        "Agg",
        force=True,
    )

    _CONFIGURED = True
