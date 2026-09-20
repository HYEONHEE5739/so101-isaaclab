"""Read-only adapter of existing Mimic generator counters."""
class GenerationProgress:
    def __init__(self, generation):
        self.generation = generation
        self.previous = (0, 0)
        self.outcome = '진행 중'

    def __call__(self):
        g = self.generation
        attempts, successes = g.num_attempts, g.num_success
        if attempts > self.previous[0]:
            self.outcome = '성공 (최근 완료 시도)' if successes > self.previous[1] else '실패 (최근 완료 시도)'
        self.previous = attempts, successes
        return dict(attempts=attempts, successes=successes, failures=g.num_failures,
                    episode_outcome=self.outcome, outcome_reason='Mimic generator 판정')
