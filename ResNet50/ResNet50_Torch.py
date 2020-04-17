from timeit import timeit

import torch
import torchvision.models as models

DEVICE_ID = 0
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
print(device)

m = models.resnet50().to(device)


def benchmark(batchsize):
    ip = torch.randn(batchsize, 3, 224, 224).to(device)

    # warmup
    m(ip)

    # benchmark
    print(timeit(lambda: m(ip), number=10))


def profile(batchsize):
    ip = torch.randn(batchsize, 3, 224, 224).to(device)

    # warmup
    m.predict(ip)

    # profile
    torch.cuda.nvtx.range_push("ResNet50 Torch")
    m(ip)
    torch.cuda.nvtx.range_pop()
