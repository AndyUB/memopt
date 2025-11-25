import torch
from typing import Any, Callable, Optional


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable,
        *args: torch.Tensor,
    ) -> Any:
        ctx.run_function = run_function
        args = args[1:]  # first arg is dummy
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        inputs = ctx.saved_tensors
        detached_inputs = []
        for inp in inputs:
            detached_inp = inp.detach()
            detached_inp.requires_grad = inp.requires_grad
            detached_inputs.append(detached_inp)

        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        torch.autograd.backward(outputs, grad_outputs)
        grads = tuple(inp.grad for inp in detached_inputs)
        return (None, None, *grads)  # None for run_function, dummy arg


def checkpoint(
    run_function: Callable,
    *args: torch.Tensor,
) -> Any:
    dummy = torch.empty((0,), requires_grad=True)
    args = (dummy, *args)
    return CheckpointFunction.apply(run_function, *args)
