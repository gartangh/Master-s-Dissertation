import torch
import torchvision.models as models

DEVICE_ID = 0
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
print(device)


def benchmark(batchsize):
    m = models.resnet50().to(device)
    ip = torch.randn(batchsize, 3, 224, 224).to(device)

    # warmup
    m(ip)

    torch.cuda.nvtx.range_push("ResNet50 Torch")
    m(ip)
    torch.cuda.nvtx.range_pop()


if __name__ == '__main__':
    benchmark(1)
