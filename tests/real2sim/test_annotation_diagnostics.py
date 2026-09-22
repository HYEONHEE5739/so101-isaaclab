from types import SimpleNamespace
import numpy as np
from soarm101_lab.real2sim.workspaces.diagnostics import GraspReport


def test_grasp_separate_conditions_not_reported_as_simultaneous(capsys):
    env = SimpleNamespace(_grasp_completed=np.array([False]))
    report = GraspReport()
    for distance, closed in [(.03, False), (.06, True)]:
        env._grasp_diagnostics = dict(distance=np.array([distance]), gripper=np.array([.2]),
            close=np.array([distance < .045]), closed=np.array([closed]), opened=np.array([True]),
            distance_limit=.045, closed_limit=.3, open_limit=.45)
        report.observe(env)
    report.report(env)
    text = capsys.readouterr().out
    assert 'Grasp FAIL' in text and '최소 30.00' in text
    assert '동시 충족: 0 frames' in text
    assert report.close == 1 and report.closed == 1


def test_wall_tolerance_boundary():
    from soarm101_lab.real2sim.workspaces.environment import inside_wall
    excess = np.array([[-.001, .00095], [0, .001], [0, .00101]])
    assert inside_wall(excess).tolist() == [True, True, False]


def test_diagnostics_remain_in_raw_log_but_not_summary(tmp_path):
    from soarm101_lab.workflow.view import LogTail
    path = tmp_path / 'process.log'
    path.write_text('[DIAGNOSTIC] Grasp FAIL\n[DIAGNOSTIC] Annotation frame 100/361\nAnnotating 5/30 (demo_12)\n❌ Skipped episode.\n')
    tail = LogTail(); lines = tail.read(path)
    assert len(lines) == 4
    assert [line for line in lines if tail.relevant(line)] == ['Annotating 5/30 (demo_12)', '❌ Skipped episode.']
    assert tail.relevant('RuntimeError: failed to load data')
