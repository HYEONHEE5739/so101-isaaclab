#!/usr/bin/env python3
"""Real/Sim worker entry point; import only the selected simulation engine."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/soarm101_lab"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default=str(ROOT / "outputs/real2sim"))
    p.add_argument('--backend', choices=('isaac', 'mujoco', 'real', 'calibration'), default='isaac')
    p.add_argument('--renderer', choices=('egl', 'osmesa', 'software-egl'), default='egl')
    if "--help" in sys.argv or "-h" in sys.argv:
        p.print_help()
        print("Also accepts Isaac AppLauncher options, e.g. --headless --device cuda:0")
        return
    if p.parse_known_args()[0].backend in ('mujoco','real','calibration'):
        p.add_argument('--headless', action='store_true')
        args = p.parse_args()
        if args.backend == 'mujoco':
            from soarm101_lab.real2sim.launcher import resolve_live_renderer
            resolved = resolve_live_renderer(args.renderer)
            if resolved != args.renderer:
                print(f'[Real2Sim] {args.renderer} unavailable; using {resolved}', flush=True)
            args.renderer = resolved
        from soarm101_lab.real2sim.runtime import Runtime
        Runtime(args, None).run()
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    # EGL/OSMesa selects MuJoCo's GL context, not SimulationApp's RTX render mode.
    # Leave Isaac's renderer at its own default (RaytracedLighting).
    del args.renderer
    args.enable_cameras = True
    started = time.monotonic()
    print('[Real2Sim][isaac] AppLauncher/Kit/RTX: START', flush=True)
    launcher = AppLauncher(args)
    print(f'[Real2Sim][isaac] AppLauncher/Kit/RTX: {time.monotonic()-started:.3f}s', flush=True)
    try:
        from soarm101_lab.real2sim.runtime import Runtime

        Runtime(args, launcher.app).run()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
