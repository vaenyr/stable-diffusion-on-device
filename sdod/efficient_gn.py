import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
import operator as op


class EfficientGN(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None, bn=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.bn = bn
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shape = input.shape
        assert shape[1] == self.num_channels
        channels_per_group = self.num_channels // self.num_groups
        spatial = reduce(op.mul, shape[2:], 1)
        input = input.reshape(shape[0] * self.num_groups, channels_per_group, spatial)

        if self.bn:
            mean = input.mean(dim=(1,2), keepdim=True)
            var = ((input-mean)**2).mean(dim=(1,2))
            input = input.permute(1, 0, 2)
            input = F.batch_norm(input, mean.squeeze(), var.squeeze(), None, None, training=False, momentum=0, eps=self.eps)
            input = input.permute(1, 0, 2)
        else:
            input = F.layer_norm(input, (channels_per_group, spatial, ), None, None, self.eps)

        input = input.reshape(shape[0], self.num_channels, *shape[2:])
        if self.affine:
            input = input * self.weight.reshape(1, self.num_channels, *tuple(1 for _ in shape[2:])) + self.bias.reshape(1, self.num_channels, *tuple(1 for _ in shape[2:]))
        return input

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
