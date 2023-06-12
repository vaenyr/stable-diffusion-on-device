import torch
import torch.nn as nn

from sdod import EfficientGN


net = nn.Sequential(
    nn.Conv2d(8, 8, 3, 1, 1),
    nn.GroupNorm(2, 8)
)

with torch.no_grad():
    torch.nn.init.normal_(net[1].weight)
    torch.nn.init.normal_(net[1].bias)

net.eval()
net.cuda()

for impl in [None, 'eff', 'ln', 'bn']:
    with torch.no_grad():
        i = torch.rand(1, 8, 2, 2, device='cuda')
        o = net(i)

        rep = EfficientGN(2, 8, impl=impl)
        rep.cuda()
        rep.load_state_dict(net[1].state_dict())
        net[1] = rep

        o2 = net(i)

        print(impl, torch.allclose(o, o2))

