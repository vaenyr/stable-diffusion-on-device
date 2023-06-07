import torch
import torch.nn as nn

from sdod import EfficientGN


net = nn.Sequential(
    nn.Conv2d(8, 8, 3, 1, 1),
    nn.GroupNorm(2, 8)
)

net.eval()
net.cuda()

with torch.no_grad():
    i = torch.rand(1, 8, 2, 2, device='cuda')
    o = net(i)

    rep = EfficientGN(2, 8, bn=True)
    rep.cuda()
    rep.load_state_dict(net[1].state_dict())
    net[1] = rep

    print(rep)

    o2 = net(i)

    print(o)
    print(o2)
    print()

    print(torch.allclose(o, o2))

