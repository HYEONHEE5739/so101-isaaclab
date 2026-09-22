from soarm101_lab.workflow.evaluation import episode_result, log_evaluation


def test_failure_and_success_rates(capsys):
    task = {'task_id': 'red_a', 'language_instruction': 'pick'}
    bad = episode_result(task, 0, 300, False, {
        'height_ok': [True], 'wall_ok': [False], 'speed_ok': [False]})
    assert bad['failure_conditions'] == ['컵 안쪽 벽', '정지 속도']
    good = episode_result(task, 1, 100, True, {
        'height_ok': [True], 'wall_ok': [True], 'speed_ok': [True]})
    assert good['failure_conditions'] == []
    log_evaluation([bad], 'red_a')
    log_evaluation([bad, good], 'red_a')
    text = capsys.readouterr().out
    assert '미충족: 컵 안쪽 벽, 정지 속도' in text
    assert '1/2 (50.0%)' in text


def test_evaluation_seed_modes(monkeypatch):
    import pytest
    from soarm101_lab.workflow.evaluation import evaluation_seed
    assert evaluation_seed({}) == 0
    assert evaluation_seed({'eval_seed': 42}) == 42
    monkeypatch.setattr('secrets.randbelow', lambda n: 123)
    assert evaluation_seed({'eval_seed_mode': 'random'}) == 123
    for value in (-1, True, '42', 2**31):
        with pytest.raises(ValueError):
            evaluation_seed({'eval_seed': value})
