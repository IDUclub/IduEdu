"""Inspect version/changelog with restored local release-tag references."""

import subprocess
import sys
from pathlib import Path

from normalize_release_tags import normalize, restore

root = Path(__file__).resolve().parents[2]
args = sys.argv[1:]
if args not in (["--noop", "version", "--print"], ["changelog"]):
    raise SystemExit("Only version inspection and changelog generation are supported")
updates = normalize(root, root / ".github/history-rewrite-tags.json")
try:
    result = subprocess.run([sys.executable, "-m", "semantic_release", *args], cwd=root, check=False)
finally:
    restore(root, updates)
raise SystemExit(result.returncode)
