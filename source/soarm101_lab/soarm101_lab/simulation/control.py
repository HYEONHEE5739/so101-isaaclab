"""Pause, single-step and cancellation shared by physics workflows. No engine imports."""
import threading


class GenerationStopped(Exception):pass


class PhysicsControl:
    def __init__(self,num_envs):
        self.condition=threading.Condition();self.paused=False;self.tokens={i:0 for i in range(num_envs)}
    def update(self,paused,single_step=False):
        with self.condition:
            self.paused=paused
            if single_step:self.tokens={i:1 for i in self.tokens}
            self.condition.notify_all()
    def before_step(self,index,stop):
        with self.condition:
            while self.paused and self.tokens[index]==0:
                if stop.is_set():raise GenerationStopped()
                self.condition.wait(.03)
            if self.paused:self.tokens[index]-=1
