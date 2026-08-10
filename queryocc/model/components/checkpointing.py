from typing import Callable

import torch


def checkpoint(fn: Callable, *args, **kwargs):
    # If gradient tracking is enabled and any input requires grad
    grad_enabled = torch.is_grad_enabled()
    input_requires_grad = any(isinstance(a, torch.Tensor) and a.requires_grad for a in args)

    if grad_enabled and input_requires_grad:
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return fn(*args)
