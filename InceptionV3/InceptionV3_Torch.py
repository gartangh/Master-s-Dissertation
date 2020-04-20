from timeit import timeit

import torch
import torchvision.models as models

DEVICE_ID = 0
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
print(device)

m = models.inception_v3(aux_logits=False).to(device)


def benchmark(batchsize):
    ip = torch.randn(batchsize, 3, 299, 299).to(device)

    # warmup
    m(ip)

    # benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    m(ip)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end))


def profile(batchsize):
    ip = torch.randn(batchsize, 3, 299, 299).to(device)

    # warmup
    m.predict(ip)

    # profile
    torch.cuda.nvtx.range_push("InceptionV3 Torch")
    m(ip)
    torch.cuda.nvtx.range_pop()
