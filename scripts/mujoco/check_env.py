#!/usr/bin/env python3
"""Independent Phase 2 smoke/measurement worker. Rendering modes use separate processes."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/soarm101_lab'))


def memory_sample(label):
    import psutil
    sample = {'stage': label, 'rss_mib': psutil.Process().memory_info().rss / 2**20,
              'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'gpu_memory': None}
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        sample['gpu_memory'] = result.stdout.strip() if result.returncode == 0 else None
        if result.returncode:
            sample['gpu_error'] = result.stderr.strip() or result.stdout.strip()
        sample['gpu_memory_scope'] = 'device-wide snapshot; not process peak or CUDA allocator peak'
    except (OSError, subprocess.TimeoutExpired) as exc:
        sample['gpu_error'] = str(exc)
    return sample


def forbidden_imports():
    return sorted(n for n in sys.modules if n.split('.')[0] in ('isaaclab', 'omni', 'pxr'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-python', help='Explicit trusted Task Python file; executed only in this worker')
    parser.add_argument('--full-workspace', action='store_true', help='Explicitly include the full supported Workspace scene')
    parser.add_argument('--scene-task-id', help='Explicit catalog scene/reset definition, independent of Python Task')
    parser.add_argument('--workspace', required=True, help='Immutable published workspace directory')
    parser.add_argument('--output', default='outputs/workflow_validation/mujoco_phase2_' + time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--renderer', choices=['none', 'egl', 'osmesa', 'software-egl'], default='none')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.frames < 1:
        parser.error('steps and frames must be positive')
    output = Path(args.output).resolve()
    # Treat the entire run directory as a new artifact, including diagnostics.
    if output.exists():
        parser.error('output exists; choose a new directory')
    if args.workspace and output.is_relative_to(Path(args.workspace).resolve()):
        parser.error('output must be outside the input workspace')
    output.mkdir(parents=True)
    report = {'schema': 'so101.mujoco-check/1', 'status': 'RUNNING', 'renderer_requested': args.renderer,
              'pid': os.getpid(), 'samples': [memory_sample('worker_before_engine_import')],
              'phase25_gate': 'NOT_RUN', 'note': 'Phase 2 checks only; no Qwen/VLA forward performed'}
    env = None
    try:
        import numpy as np
        from soarm101_lab.simulation.mujoco_backend import MujocoBackend
        task = context = None
        if args.task_python:
            import runpy
            task=runpy.run_path(str(Path(args.task_python).resolve()))
            from soarm101_lab.real2sim.workspaces.tasks import validate_task_module
            validate_task_module(task)
        from soarm101_lab.real2sim.workspaces.package import load
        from soarm101_lab.real2sim.workspaces.tasks import catalog
        workspace=load(args.workspace)
        definition=None
        if args.scene_task_id:
            definition=next((d for d in catalog(workspace) if d['task_id']==args.scene_task_id),None)
            if definition is None:raise ValueError('Unknown explicit scene task: '+args.scene_task_id)
        env = MujocoBackend(workspace=workspace, render_backend=args.renderer,
                            width=args.width, height=args.height,
                            task_definition=definition, full_workspace=args.full_workspace)

        report['mujoco_version'] = env.mj.__version__
        report['samples'].append(memory_sample('physics_loaded_renderer_not_created'))
        initial = env.reset(seed=args.seed)
        saved = env.get_initial_state()
        if task is not None:
            from soarm101_lab.simulation.task_context import TaskContext
            context=TaskContext(env)
            context.validate_bindings(task)
            task['setup'](context)
            def sample_task():
                observation=task['observe'](context)
                outcome=task['success'](context)
                if not isinstance(outcome,(bool,np.bool_)):
                    raise ValueError('Task success(ctx) must return bool')
                return {'observation':observation, 'success':bool(outcome)}
            report['task_python']={'task_id':task['TASK_ID'], 'instruction':task['INSTRUCTION'],
                'initial':sample_task(), 'robot_identity_scope':'Optional asset reference/manifest hash equality; no hardware identity certification'}
            for _ in range(args.steps):env.step(env.home)
            report['task_python']['final']=sample_task()
            report['scene_metadata']=env.metadata
            report['forbidden_imports']=forbidden_imports()
            assert not report['forbidden_imports']
            report['status']='PASSED'
            return

        report['initial_eef_pose_workspace'] = initial['eef_pose'].tolist()
        report['initial_object_pose_workspace'] = initial['object_poses'][env.metadata['pick']].tolist()
        assert env.model.nu == 6
        assert env.model.ncam == (len(env.metadata.get('cameras',{})) or 1)
        target = env.home.copy()
        target[0] += .1
        begin = time.perf_counter()
        for _ in range(args.steps):
            state = env.step(target)
        report['control_steps_per_second'] = args.steps / (time.perf_counter() - begin)
        report['simulated_seconds'] = state['timestamp_s']
        report['measured_joint_rad'] = state['joint_pos'].tolist()
        report['commanded_joint_rad'] = state['joint_targets'].tolist()
        report['target_max_abs_error_rad'] = float(np.max(np.abs(state['joint_pos'] - target)))
        report['object_speed_m_s'] = env.success_details()['speed_m_s']
        assert np.max(np.abs(state['joint_pos'] - initial['joint_pos'])) > .01, 'Robot did not move'
        assert not env.is_success(), 'Initial object unexpectedly in target'
        report['samples'].append(memory_sample('physics_after_steps'))
        if args.renderer != 'none':
            from PIL import Image
            begin = time.perf_counter()
            frame = env.get_rgb()
            report['first_frame_latency_s'] = time.perf_counter() - begin
            assert frame.shape == (args.height, args.width, 3) and frame.dtype == np.uint8
            assert frame.std() > 1, 'Blank camera image'
            Image.fromarray(frame).save(output / 'rgb.png')
            report['samples'].append(memory_sample('renderer_after_first_frame'))
            durations = []
            for _ in range(args.frames):
                env.step(target)
                begin = time.perf_counter()
                env.get_rgb()
                durations.append(time.perf_counter() - begin)
            report['camera_render_fps'] = len(durations) / sum(durations)
            report['camera_render_latency_ms_median'] = float(np.median(durations) * 1000)
            report['camera_render_latency_ms_p95'] = float(np.percentile(durations, 95) * 1000)
            report['camera_render_scope'] = 'render+readback only; physics, saving and policy excluded'
            report['render_context'] = env.metadata['render_context']
            report['samples'].append(memory_sample('renderer_steady_after_frames'))
        # Positive success check is a geometric placement probe, not a solved manipulation rollout.
        cup = env.workspace['profile']['objects'][env.workspace['task']['targets'][env.metadata['target']]]
        size = env.workspace['profile']['objects'][env.metadata['pick']]['dimensions_m']
        pose = env.get_object_pose(env.metadata['target']).copy()
        pose[:3, 3] += pose[:3, :3] @ np.array([0, 0, -cup['dimensions_m'][2] / 2 + cup['wall_m'] + size[2] / 2 + .001])
        env.set_object_pose(env.metadata['pick'], pose)
        for _ in range(100):
            env.step(target)
        report['placement_probe'] = env.success_details()
        assert env.is_success(), 'Placed cube failed contained-and-settled check'
        env.restore_state(saved)
        np.testing.assert_allclose(env.get_state()['joint_pos'], initial['joint_pos'], atol=1e-12)
        np.testing.assert_allclose(env.get_object_pose(env.metadata['pick']), initial['object_poses'][env.metadata['pick']], atol=1e-12)
        report['reset_restore_verified'] = True
        report['forbidden_imports'] = forbidden_imports()
        assert not report['forbidden_imports']
        report['scene_metadata'] = env.metadata
        report['status'] = 'PASSED'
    except BaseException as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', forbidden_imports=forbidden_imports())
        raise
    finally:
        if env is not None:
            env.close()
        report['environment_closed']=env is None or env.closed
        report['samples'].append(memory_sample('worker_after_close'))
        (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda v: v.tolist() if hasattr(v,'tolist') else str(v)))
        print(json.dumps({'status': report['status'], 'report': str(output / 'report.json'),
                          'phase25_gate': report['phase25_gate']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
