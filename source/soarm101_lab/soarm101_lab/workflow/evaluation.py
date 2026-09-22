"""Workspace policy evaluation. Real motion is opt-in and never used by tests."""
import json
import time
from pathlib import Path
from ..real2sim.storage import atomic
from ..real2sim.workspaces.package import load, metadata
from ..real2sim.workspaces.demo import coordinate_contract
from .policy import load_policy, observation



def evaluation_seed(options):
    mode = options.get('eval_seed_mode', 'fixed')
    if mode == 'random':
        import secrets
        return secrets.randbelow(2**31)
    if mode != 'fixed':
        raise ValueError('eval_seed_mode must be fixed or random')
    seed = options.get('eval_seed', 0)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('eval_seed must be an integer in [0, 2147483647]')
    return seed


def episode_result(definition, episode, steps, success, details):
    checks = {name: bool(details[key][0]) for name, key in
              [('높이', 'height_ok'), ('컵 안쪽 벽', 'wall_ok'), ('정지 속도', 'speed_ok')]}
    failed = [name for name, passed in checks.items() if not passed]
    return dict(task_id=definition['task_id'], language_instruction=definition['language_instruction'],
                episode=episode, steps=steps, success=success, checks=checks,
                failure_conditions=failed,
                outcome_reason='컵 안 정지 조건 충족' if success else '최대 step 도달 · 미충족: ' + ', '.join(failed))


def log_evaluation(rows, task_id):
    selected = [r for r in rows if r['task_id'] == task_id]
    last = rows[-1]
    wins = sum(r['success'] for r in selected)
    total_wins = sum(r['success'] for r in rows)
    print(f"[WORKFLOW] Sim 평가 {task_id} Episode {last['episode'] + 1}: "
          f"{'성공' if last['success'] else '실패'} · {last['steps']} steps · {last['outcome_reason']}", flush=True)
    print(f"[WORKFLOW] Task 성공률 {wins}/{len(selected)} ({100*wins/len(selected):.1f}%) · "
          f"전체 누적 {total_wins}/{len(rows)} ({100*total_wins/len(rows):.1f}%)", flush=True)


def evaluate(request):
    from .presentation import current, attach
    bridge = current()
    stage, opts = request['stage'], request['options']
    if stage == 'real_eval' and not opts.get('enable_motion'):
        raise ValueError('Real motion must be explicitly enabled')
    app = None
    if stage == 'sim_eval':
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=opts.get('headless', True), enable_cameras=True).app
    try:
        import torch
        w = load(request['workspace']); device = opts.get('device', 'cuda:0')
        artifact = request['input']; path = artifact['path']
        if artifact['type'] == 'hf_policy':
            remote = json.loads(Path(path).read_text())
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo_id=remote['repo_id'], revision=remote['commit'])
        policy, pre, post = load_policy(path, coordinate_contract(w), device)
        from .tasks import catalog, selected, validate, configured
        definitions = catalog(w) if opts.get('task_scope') == 'all' else opts.get('tasks') or [selected(w)]
        definitions = [validate(t, w) for t in definitions]
        if not definitions or len({t['task_id'] for t in definitions}) != len(definitions):
            raise ValueError('Select distinct evaluation tasks')
        rows = []
        used_seed = None
        if stage == 'sim_eval':
            from isaaclab.envs import ManagerBasedEnv
            from ..real2sim.workspaces.environment import make_config, task_success_details
            used_seed = evaluation_seed(opts)
            cfg = make_config(w['root'], device, task_definition=definitions[0])
            cfg.seed = used_seed
            cfg.events.reset_episode.params['randomization_seed'] = used_seed
            print(f"[WORKFLOW] Sim 평가 seed={used_seed} ({opts.get('eval_seed_mode', 'fixed')})", flush=True)
            atomic(str(request['output']) + '.seed.json', {'seed': used_seed, 'mode': opts.get('eval_seed_mode', 'fixed')})
            env = ManagerBasedEnv(cfg=cfg)
            attach(env)
            try:
                for definition, ep in [(d, e) for d in definitions for e in range(int(opts.get('episodes', 1)))]:
                    task = definition['language_instruction']
                    print(f"[WORKFLOW] Sim 평가 시작 · {definition['task_id']} · Episode {ep + 1}/{int(opts.get('episodes', 1))}", flush=True)
                    env._workflow_task_workspace = configured(w, definition)
                    if bridge:
                        bridge.details.update(task_id=definition['task_id'], language_instruction=task, episode_index=ep + 1, episode_total=int(opts.get('episodes', 1)), episode_outcome='진행 중', outcome_reason='')
                    for field in ('_grasp_completed', '_gripper_opened_once'):
                        if hasattr(env, field): getattr(env, field).zero_()
                    obs, _ = env.reset(); policy.reset(); success = False
                    for step in range(int(opts.get('max_steps', 300))):
                        p = obs['policy']
                        batch = observation(p['joint_pos'][0], p['side_cam'][0], p['wrist_cam'][0], task, device)
                        with torch.inference_mode():
                            action = post(policy.select_action(pre(batch))).reshape(1, 6)
                        if not torch.isfinite(action).all():
                            raise ValueError('Nonfinite policy action')
                        obs, _ = env.step(action.to(env.device))
                        result, details = task_success_details(env, w)
                        success = bool(result[0])
                        if success:
                            break
                    rows.append(episode_result(definition, ep, step + 1, success, details))
                    log_evaluation(rows, definition['task_id'])
                    from ..real2sim.workspaces.diagnostics import report_place
                    report_place(env, w)
                    if bridge:
                        bridge.details.update(episode_outcome='성공' if success else '실패', outcome_reason=rows[-1]['outcome_reason']); bridge.status()
            finally:
                env.close()
        else:
            from ..real2sim.hardware import Follower
            from ..real2sim.cameras import CameraReader
            from ..representation import real_observation, real_action
            follower = Follower(opts['port'], opts['follower_id'], opts.get('calibration_dir'))
            cameras = {}
            try:
                for view in ('side', 'wrist'):
                    width, height = w['profile']['cameras'][view]['resolution']
                    cameras[view] = CameraReader(opts[view + '_device'], width, height)
                    cameras[view].start()
                deadline = time.monotonic() + 10
                while not all(c.nearest() is not None for c in cameras.values()):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Cameras did not deliver frames; no motion started')
                    time.sleep(.02)
                follower.connect(); follower.activate(); policy.reset()
                for definition, ep in [(d, e) for d in definitions for e in range(int(opts.get('episodes', 1)))]:
                    task = definition['language_instruction']; policy.reset()
                    if bridge:
                        bridge.details.update(task_id=definition['task_id'], language_instruction=task, episode_index=ep + 1, episode_total=int(opts.get('episodes', 1)), episode_outcome='진행 중', outcome_reason='')
                    for step in range(int(opts.get('max_steps', 300))):
                        if bridge:
                            bridge.before_step()
                        begin = time.monotonic()
                        native, _ = follower.read()
                        frames = {v: c.nearest() for v, c in cameras.items()}
                        if any(time.monotonic_ns() - f['read_end']['monotonic_ns'] > 500_000_000 for f in frames.values()):
                            raise RuntimeError('Stale real camera frame')
                        if bridge:
                            bridge.steps = step
                            bridge.publish({'real_' + v: f['rgb'] for v, f in frames.items()})
                        q = real_observation({n + '.pos': v for n, v in native.items()})
                        batch = observation(q, frames['side']['rgb'], frames['wrist']['rgb'], task, device)
                        with torch.inference_mode():
                            action = post(policy.select_action(pre(batch))).reshape(6).detach().cpu().numpy()
                        command = real_action(action)
                        follower.send({n[:-4]: v for n, v in command.items()})
                        rows.append({'task_id': definition['task_id'], 'language_instruction': task, 'episode': ep, 'step': step, 'model_observation': q.tolist(), 'model_action': action.tolist()})
                        time.sleep(max(0, 1 / int(opts.get('fps', 30)) - (time.monotonic() - begin)))
            finally:
                try:
                    follower.close()
                finally:
                    for camera in cameras.values():
                        camera.close()
        from .tasks import evaluation_metrics
        if stage == 'sim_eval':
            for definition in definitions:
                subset = [r for r in rows if r['task_id'] == definition['task_id']]
                wins = sum(r['success'] for r in subset)
                print(f"[WORKFLOW] 평가 완료 · {definition['task_id']} · 성공 {wins}/{len(subset)} · 성공률 {100*wins/len(subset):.1f}%", flush=True)
        atomic(request['output'], {'status': 'SUCCEEDED', 'mode': stage, 'workspace': metadata(w),
               'policy_artifact': artifact['id'], 'task_definitions': definitions, 'evaluation_seed': used_seed,
               'metrics_per_task': evaluation_metrics(definitions, rows, stage == 'sim_eval'), 'episodes' if stage == 'sim_eval' else 'steps': rows,
               'task_success': 'simulation geometric condition' if stage == 'sim_eval' else 'unmeasured; human review required'})
    except Exception:
        # Report before Isaac shutdown, which can itself take time after setup failure.
        import traceback
        traceback.print_exc()
        raise
    finally:
        if bridge:
            bridge.close()
        if app:
            app.close()
