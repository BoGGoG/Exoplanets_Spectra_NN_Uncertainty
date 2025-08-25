import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def compute_cosine_decay(
    step: int,
    warmup: int,
    total_steps: int,
    steps_per_cycle: int,
    cycle_stretch: float = 1.0,
    decay: float = 1.0,
    min_lr: float = 1e-8,
) -> float:
    """
    Inspired by [HuggingFace's implementation](https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/optimization.py#L178-L219) of a cosine decay scheduler with restarts and decay.

    Compute the learning rate using a cosine decay schedule with restarts and decay.

    Args:
        step (int): Current training step.
        warmup (int): Number of warmup steps.
        total_steps (int): Total training steps.
        steps_per_cycle (int): Number of steps in one cycle.
        cycle_stretch (float): How much longer to make each consecutive cycle.
        decay (float): Decay factor for each cycle.

    Returns:
        float: Computed learning rate.
    """
    if step < warmup:
        return step / max(1, warmup)

    progress = (step - warmup) / max(1, total_steps - warmup)
    if progress >= 1.0:
        return 0.0

    remaining_steps = step - warmup
    cycle_idx = 0
    steps_this_cycle = steps_per_cycle

    while remaining_steps >= steps_this_cycle:
        remaining_steps -= steps_this_cycle
        cycle_idx += 1
        steps_this_cycle = int(steps_per_cycle * (cycle_stretch**cycle_idx))

    cycle_progress = remaining_steps / steps_this_cycle
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))

    return max(min_lr, cosine_decay * (decay**cycle_idx))


def cosine_decay_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    steps_per_cycle: int,
    last_epoch: int = -1,
    decay_factor: float = 1.0,
    cycle_stretch: float = 1.0,
    min_lr: float = 1e-8,
) -> LambdaLR:
    """
    Inspired by [HuggingFace's implementation](https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/optimization.py#L178-L219) of a cosine decay scheduler with restarts and decay.

    Create a LambdaLR scheduler with a cosine decay schedule with restarts and decay.

    Args:
        optimizer (Optimizer): Optimizer for which to schedule the learning rate.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of training steps.
        steps_per_cycle (int): Number of steps in one cycle.
        last_epoch (int, optional): Index of the last epoch. Defaults to -1.
        decay_factor (float, optional): Decay factor for each cycle. Defaults to 1.0.
        cycle_stretch (float, optional): How much longer to make each consecutive cycle. Defaults to 1.0.

    Returns:
        LambdaLR: Learning rate scheduler.
    """
    lr_lambda = partial(
        compute_cosine_decay,
        warmup=warmup_steps,
        total_steps=total_steps,
        steps_per_cycle=steps_per_cycle,
        cycle_stretch=cycle_stretch,
        decay=decay_factor,
        min_lr=min_lr,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
