"""Step-wise LambdaLR: linear decay (epoch 1), then cosine restarts.

The paper omits phase endpoints and restart-period values. The arguments
below are EXPLICIT PROVISIONAL DEFAULTS, not verified original-run values.
"""
from math import cos, pi


def lr_multiplier(step, steps_per_epoch, linear_end=0.5, restart_epochs=4, minimum=0.01):
    if steps_per_epoch < 1 or restart_epochs < 1:
        raise ValueError("Invalid epoch/restart length.")
    if not 0 < minimum <= linear_end <= 1:
        raise ValueError("Invalid learning-rate factors.")
    if step < steps_per_epoch:
        return 1.0 - (1.0 - linear_end) * max(step, 0) / steps_per_epoch
    period = steps_per_epoch * restart_epochs
    phase = ((step - steps_per_epoch) % period) / period
    return minimum + (linear_end - minimum) * (1 + cos(pi * phase)) / 2
