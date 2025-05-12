from stable_baselines3.common.utils import constant_fn

constant_scedule = constant_fn

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func

class LinearParameter:
    def __init__(self, start=1.0, end=0.1, num_steps=10):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.step_decrement = (start - end) / float(num_steps)
        self.value = start
        self.current_step = 0

    def decrease(self):
        """Decreases the parameter linearly for one step and ensures it doesn't go below the minimum."""
        if self.current_step < self.num_steps:
            self.value -= self.step_decrement
            self.value = max(self.value, self.end)
            self.current_step += 1