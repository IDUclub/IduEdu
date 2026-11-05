"""Compatibility for ``manuscript_cus/make_source_bundle.py``; the figures are drawn by ``pipeline/figures.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from figures import COLUMN_IN  # noqa: E402,F401
