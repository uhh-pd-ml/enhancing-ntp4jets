"""Callbacks for monitoring the model.

Based on https://github.com/mattcleigh/mltools/blob/78037c9cdd6ba3ba71cf8e5f5d0af4920f74c5e4/mltools/lightning_utils.py

MIT License

Copyright (c) 2023 Matthew Leigh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from lightning import Callback, LightningModule, Trainer
from torch import nn
from torch.optim import Optimizer


def gradient_norm(model) -> float:
    """Return the total norm of the gradients of a model.

    The strange logic is to avoid upscaling the norm when using mixed precision.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_data = p.grad.detach().data.square().sum()
            if total_norm == 0:
                total_norm = grad_data
            else:
                total_norm += grad_data
    if total_norm == 0:
        return 0
    return total_norm.sqrt().item()


def weight_norm(model) -> float:
    """Return the total norm of the weights of a model.

    The strange logic is to avoid upscaling the norm when using mixed precision.
    """
    total_norm = 0
    for p in model.parameters():
        if p.data is not None:
            weight_data = p.data.square().sum()
            if total_norm == 0:
                total_norm = weight_data
            else:
                total_norm += weight_data
    if total_norm == 0:
        return 0
    return total_norm.sqrt().item()


def get_submodules(module: nn.Module, depth: int = 1, prefix="") -> list:
    """Return a list of all of the base modules in a network."""
    modules = []
    if depth == 0 or not list(module.children()):
        return [(prefix, module)]
    for n, child in module.named_children():
        subname = prefix + ("." if prefix else "") + n
        # skip if is not a torch module
        if not isinstance(child, nn.Module):
            continue
        # skip if name is "criterion"
        if n == "criterion":
            continue
        modules.extend(get_submodules(child, depth - 1, subname))
    return modules


class LogGradNorm(Callback):
    """Logs the gradient norm."""

    def __init__(self, logging_interval: int = 1, depth: int = -1):
        self.logging_interval = logging_interval
        self.depth = depth

    def on_before_optimizer_step(
        self, _trainer: Trainer, pl_module: LightningModule, _optimizer: Optimizer
    ):
        if pl_module.global_step % self.logging_interval == 0:
            sub_modules = get_submodules(pl_module, self.depth)
            for subname, module in sub_modules:
                grad = gradient_norm(module)
                if grad > 0:
                    self.log("grad/" + subname, grad)


class LogWeightNorm(Callback):
    """Logs the weight norm."""

    def __init__(self, logging_interval: int = 1, depth: int = -1):
        self.logging_interval = logging_interval
        self.depth = depth

    def on_before_optimizer_step(
        self, _trainer: Trainer, pl_module: LightningModule, _optimizer: Optimizer
    ):
        if pl_module.global_step % self.logging_interval == 0:
            sub_modules = get_submodules(pl_module, self.depth)
            for subname, module in sub_modules:
                weight = weight_norm(module)
                self.log("weight/" + subname, weight)
