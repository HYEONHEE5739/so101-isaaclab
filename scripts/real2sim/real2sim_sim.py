#!/usr/bin/env python3
"""Isaac entry point; all runtime operations are requested through the PyQt UI."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/soarm101_lab"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default=str(ROOT / "outputs/real2sim"))
    if "--help" in sys.argv or "-h" in sys.argv:
        p.print_help()
        print("Also accepts Isaac AppLauncher options, e.g. --headless --device cuda:0")
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.enable_cameras = True
    launcher = AppLauncher(args)
    try:
        from soarm101_lab.real2sim.runtime import Runtime

        Runtime(args, launcher.app).run()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
