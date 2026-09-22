"""Read-only annotation diagnostics. Thresholds come from the actual predicates."""
def report_place(env, workspace):
    from .environment import task_success_details
    result, d = task_success_details(env, workspace)
    value = lambda key: float(d[key][0].item())
    flag = lambda key: 'PASS' if bool(d[key][0]) else 'FAIL'
    print(f"[DIAGNOSTIC] Place {d['pick']} -> {d['target']} ({'PASS' if bool(result[0]) else 'FAIL'})", flush=True)
    print(f"[DIAGNOSTIC] 높이 {flag('height_ok')}: 꼭짓점 Z [{value('zmin')*1000:.2f}, {value('zmax')*1000:.2f}] mm; 컵 로컬 허용 ({d['lower']*1000:.2f}, {d['upper']*1000:.2f}) mm", flush=True)
    print(f"[DIAGNOSTIC] 안쪽 벽 {flag('wall_ok')}: 최대 반경 초과 {value('radial_excess')*1000:.2f} mm (허용 {d['wall_tolerance']*1000:.2f} mm 이하)", flush=True)
    print(f"[DIAGNOSTIC] 속도 {flag('speed_ok')}: {value('speed')*1000:.2f} mm/s < {d['speed_limit']*1000:.2f} mm/s", flush=True)


class GraspReport:
    def __init__(self):
        self.count = 0
        self.close = self.closed = self.opened = self.simultaneous = 0
        self.minimum = float('inf')
        self.last = None
        self.stable_seen = False
        self.stable_best = 0.0

    def observe(self, env):
        stable = getattr(env, '_stable_grasp_diagnostics', None)
        if stable:
            self.last_stable = stable[0]
            self.stable_seen |= bool(getattr(env, '_grasp_completed')[0])
            self.stable_best = max(self.stable_best, stable[0]['hold_seconds'])
            return
        d = getattr(env, '_grasp_diagnostics', None)
        if d is None: return
        self.last = d
        self.count += 1
        self.minimum = min(self.minimum, float(d['distance'][0]))
        for key in ('close', 'closed', 'opened'):
            setattr(self, key, getattr(self, key) + int(d[key][0]))
        self.simultaneous += int(d['close'][0] & d['closed'][0] & d['opened'][0])

    def report(self, env):
        if hasattr(self, 'last_stable'):
            d = self.last_stable
            print(f"[DIAGNOSTIC] Stable grasp {'PASS' if self.stable_seen else 'FAIL'} · 최대 연속 유지 {self.stable_best:.3f}s / 필요 {d['required_seconds']:.3f}s", flush=True)
            print(f"[DIAGNOSTIC] 양쪽 접촉: 고정={d['left_contact']} ({d['left_force_n']:.5f}N), 이동={d['right_contact']} ({d['right_force_n']:.5f}N); 동반 이동={d['moving']} · EE {d['ee_motion_m']*1000:.2f}mm / 물체 {d['object_motion_m']*1000:.2f}mm", flush=True)
            print("[DIAGNOSTIC] Grasp는 확인 시점에 latch됨. 이후 유지 상실은 정상 release 또는 낙하일 수 있으며 place 판정과 별도입니다.", flush=True)
            return
        if self.last is None:
            print('[DIAGNOSTIC] Grasp 진단 없음', flush=True); return
        d = self.last
        completed = getattr(env, '_grasp_completed', None)
        passed = completed is not None and bool(completed[0])
        print(f"[DIAGNOSTIC] Grasp {'PASS' if passed else 'FAIL'}: episode 측정 {self.count} frames", flush=True)
        print(f"[DIAGNOSTIC] TCP–물체 거리 < {d['distance_limit']*1000:.1f} mm: {self.close} frames; 최소 {self.minimum*1000:.2f} mm / 마지막 {float(d['distance'][0])*1000:.2f} mm", flush=True)
        print(f"[DIAGNOSTIC] 그리퍼 열림 이력 (q > {d['open_limit']:.3f} rad): {'있음' if self.opened else '없음'}; 닫힘 (q < {d['closed_limit']:.3f} rad): {self.closed} frames; 마지막 q={float(d['gripper'][0]):.3f} rad", flush=True)
        print(f"[DIAGNOSTIC] Grasp 세 조건 동시 충족: {self.simultaneous} frames (거리/닫힘이 다른 시점에 통과한 것만으로는 성공하지 않음)", flush=True)
