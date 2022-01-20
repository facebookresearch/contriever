# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist

class Gather(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.tensor):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(*grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def gather(x: torch.tensor):
    x_gather = Gather.apply(x)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather

@torch.no_grad()
def gather_nograd(x: torch.tensor):
    x_gather = [torch.ones_like(x)
        for _ in range(dist.get_world_size())]
    dist.all_gather(x_gather, x, async_op=False)

    x_gather = torch.cat(x_gather, dim=0)
    return x_gather

@torch.no_grad()
def varsize_gather_nograd(x: torch.Tensor):
    """gather tensors of different sizes along the first dimension"""

    #determine max size
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(allsizes, size)
    max_size = max([size.cpu().max() for size in allsizes])

    padded = torch.empty(
                max_size, 
                *x.shape[1:],
                dtype=x.dtype,
                device=x.device
            )
    padded[:x.shape[0]] = x
    output = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(output, padded)

    output = [tensor[:allsizes[k]] for k, tensor in enumerate(output)]
    output = torch.cat(output, dim=0)

    return output

def is_main():
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0