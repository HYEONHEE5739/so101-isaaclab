"""Mimic generation options; both engines use the same screen and formatter."""
from .base import StageScreen
from ..settings import fold, ResumeDataSettings
from ....simulation.ik import validate_solver, validate_failure_policy


class DataGenScreen(StageScreen):
    action = '데이터 생성 시작'
    basic_keys = {'num_successful_demos', 'num_envs', 'ik_solver', 'ik_failure_policy'}
    basic_order = ('num_successful_demos', 'num_envs', 'ik_solver', 'ik_failure_policy')

    def build_inputs(self):
        advanced = fold(self.settings_layout, '고급 생성 설정 · Resume Data')
        self.resume = ResumeDataSettings(self.update_ready)
        self.resume_data = self.resume.path
        advanced.addRow(self.resume)

    def collect_options(self):
        opts = super().collect_options()
        opts.pop('resume_data', None)
        opts['ik_solver'] = validate_solver(opts.get('ik_solver', 'scipy_ls'))
        opts['ik_failure_policy'] = validate_failure_policy(opts.get('ik_failure_policy', 'strict'))
        if self.resume_data.text().strip():
            opts['resume_data'] = self.resume_data.text().strip()
        return opts
