# -*- coding: utf-8 -*-
"""
Adapted from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L124
"""

# changed by hzhang12 on 2025-07-17

import torch
from step1x3d_geometry.utils.typing import *


def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
    use_deepspeed: bool = False,
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    :param use_deepspeed: if True, use deepspeed
    """
    if flag:
        if use_deepspeed:
            import deepspeed

            return deepspeed.checkpointing.checkpoint(func, *inputs)

        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *output_grads):
        # Detach and prepare inputs: only floating inputs require gradients
        detached_inputs = []
        for x in ctx.input_tensors:
            x_det = x.detach()
            if torch.is_floating_point(x_det):
                x_det = x_det.requires_grad_(True)
            detached_inputs.append(x_det)
        # Run forward with gradients enabled
        with torch.enable_grad():
            shallow = [x.view_as(x) for x in detached_inputs]
            out = ctx.run_function(*shallow)
        # Normalize outputs and grads as sequences
        if isinstance(out, torch.Tensor):
            outputs = (out,)
        else:
            outputs = tuple(out)
        grads_out = tuple(output_grads)
        # Filter only floating outputs
        f_outputs = []
        f_grads = []
        for o, g in zip(outputs, grads_out):
            if torch.is_floating_point(o):
                f_outputs.append(o)
                f_grads.append(g)
        # Prepare inputs requiring grads
        float_inputs = [x for x in detached_inputs if x.requires_grad]
        # Compute gradients
        grads = torch.autograd.grad(
            f_outputs,
            float_inputs + ctx.input_params,
            f_grads,
            allow_unused=True,
        )
        # Split input and param grads
        num_f = len(float_inputs)
        input_grads = grads[:num_f]
        param_grads = grads[num_f:]
        # Map back to all inputs
        full_grads = []
        it = iter(input_grads)
        for x in detached_inputs:
            if x.requires_grad:
                full_grads.append(next(it))
            else:
                full_grads.append(None)
        # Cleanup
        del ctx.input_tensors, ctx.input_params
        # Return None for run_function and length, then input and param grads
        return (None, None) + tuple(full_grads) + tuple(param_grads)
