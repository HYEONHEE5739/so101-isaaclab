#!/usr/bin/env python3
"""Desktop entry point; use a GUI-only venv, without Isaac or LeRobot imports."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/soarm101_lab"))
from soarm101_lab.real2sim.ui import main

if __name__ == "__main__":
    main()
