import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
import operator as op


class EfficientGNFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_groups, weight=None, bias=None, eps=1e-5):
        return F.group_norm(input, num_groups, weight, bias, eps)

    @staticmethod
    @torch.onnx.symbolic_helper.quantized_args(True, False, False, False)
    @torch.onnx.symbolic_helper.parse_args("v", "i", "v", "v", "f")
    def symbolic(g: torch.Graph, input, num_groups, weight, bias, eps):
        if weight is None or torch.onnx.symbolic_helper._is_none(weight):
            assert bias is None or torch.onnx.symbolic_helper._is_none(bias)
            ret = g.op('sdod::ParameterlessGroupNorm', input, num_groups_i=num_groups, eps_f=eps)
        else:
            assert bias is not None and not torch.onnx.symbolic_helper._is_none(bias)
            ret = g.op('sdod::GroupNorm', input, weight, bias, num_groups_i=num_groups, eps_f=eps)

        ret.setType(input.type())
        return ret


def efficient_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    return EfficientGNFun.apply(input, num_groups, weight, bias, eps)


class EfficientGN(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None, impl=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        if impl not in [None, 'eff', 'ln', 'bn']:
            raise ValueError('EfficientGN impl parameter should be one of None, "eff", "ln" or "bn"')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.impl = impl
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

        if self.impl is None:
            return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        elif self.impl == 'eff':
            return efficient_group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        elif self.impl == 'bn':
            input = input.reshape(shape[0] * self.num_groups, channels_per_group, spatial)
            input = input.permute(1, 0, 2)
            input = F.batch_norm(input, torch.zeros(shape[0] * self.num_groups, dtype=input.dtype, device=input.device), torch.ones(shape[0] * self.num_groups, dtype=input.dtype, device=input.device), None, None, training=False, momentum=0, eps=self.eps)
            input = input.permute(1, 0, 2)
            input = input.reshape(shape[0], self.num_channels, *shape[2:])
        elif self.impl == 'ln':
            input = input.reshape(shape[0], self.num_groups, channels_per_group*spatial)
            input = F.layer_norm(input, (channels_per_group*spatial, ), torch.ones_like(input[0,0]), torch.zeros_like(input[0,0]), eps=self.eps)
            input = input.reshape(shape[0], self.num_channels, *shape[2:])
        else:
            raise NotImplementedError(self.impl)

        # if self.impl not in [None, 'eff'] and self.affine:
        #     input = input * self.weight.reshape(1, self.num_channels, *tuple(1 for _ in shape[2:])) + self.bias.reshape(1, self.num_channels, *tuple(1 for _ in shape[2:]))
        return input

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
