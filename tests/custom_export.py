import io
import torch
import onnx

from sdod import EfficientGN

net = EfficientGN(2, 8)
net.cuda()

with torch.no_grad():
    with torch.jit.optimized_execution(True):
        buff = io.BytesIO()
        torch.onnx.export(net, torch.rand(1, 8, 32, 32, device='cuda'), buff)

        buff.seek(0)
        g = onnx.load(buff).graph

print(onnx.helper.printable_graph(g))


net = EfficientGN(2, 8, impl='eff')
net.cuda()

with torch.no_grad():
    with torch.jit.optimized_execution(True):
        buff = io.BytesIO()
        torch.onnx.export(net, torch.rand(1, 8, 32, 32, device='cuda'), buff, custom_opsets={"sdod": 1})

        buff.seek(0)
        g = onnx.load(buff).graph



print(onnx.helper.printable_graph(g))

buff.seek(0)
with open('../stable-diffusion/onnx/gn.onnx', 'wb') as f:
    f.write(buff.getvalue())
