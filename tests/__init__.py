"""Test package and early numerical-dependency bootstrap for pytest-cov."""

import os

import pytest


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_load_initial_conftests(args):
    """Initialize third-party extensions before pytest-cov enables tracing.

    Conftest is loaded after pytest-cov starts, which is too late for some
    Windows NumPy/Numba extensions. This wrapper runs before that plugin;
    iduedu itself is imported later and remains fully measured.
    """
    if "--no-cov" not in args and any(arg == "--cov" or arg.startswith("--cov=") for arg in args):
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        os.environ.setdefault("NUMBA_JIT_COVERAGE", "1")
        import geopandas  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        import numpy.fft  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        import scipy.sparse.csgraph  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        from numba.typed import Dict, List  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    yield
