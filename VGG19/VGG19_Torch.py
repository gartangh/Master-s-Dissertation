import torch
import torchvision.models as models

DEVICE_ID = 0
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
print(device)

m = models.vgg19_bn().to(device)


def benchmark_pytorch(batchsize):
    ip = torch.randn(batchsize, 3, 224, 224).to(device)

    # warmup
    torch.cuda.nvtx.range_push("VGG19 PyTorch")
    m(ip)
    torch.cuda.nvtx.range_pop()

    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda.nvtx.range_push("VGG19 PyTorch")
        m(ip)
        torch.cuda.nvtx.range_pop()
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        print(start.elapsed_time(end))
