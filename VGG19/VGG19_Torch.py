import torch
import torchvision.models as models

DEVICE_ID = 6
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
print(device)


def benchmark(batchsize):
    m = models.vgg19_bn().to(device)
    ip = torch.randn(batchsize, 3, 299, 299).to(device)

    # warmup
    m(ip)

    torch.cuda.nvtx.range_push("Profiling")
    m(ip)
    torch.cuda.nvtx.range_pop()


if __name__ == '__main__':
    benchmark(4)
