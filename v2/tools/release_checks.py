"""Release gates for the v2 Space: tests, site-data freshness, staged-tree public scan."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKS = (
    [sys.executable, "-m", "pytest", "tests_v2", "-q", "-p", "no:cacheprovider"],
    [sys.executable, "-m", "pramaan_v2.build.site_data", "--check"],
    [sys.executable, "v2/tools/deploy_space.py"],
)


def main() -> int:
    for command in CHECKS:
        print("$ " + " ".join(command[1:]), flush=True)
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            print("FAIL", flush=True)
            return 1
    print("PASS: all v2 release checks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
