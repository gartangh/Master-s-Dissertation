import torch
import torchvision.models as models

DEVICE_ID = 0
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
print(device)

gm = models.resnext50_32x4d().to(device)
gm.eval()


def benchmark_pytorch(batchsize):
    gip = torch.randn(batchsize, 3, 224, 224).to(device)

    # warm-up
    torch.cuda.nvtx.range_push("ResNeXt50 PyTorch")
    gm(gip)
    torch.cuda.nvtx.range_pop()

    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda.nvtx.range_push("ResNeXt PyTorch")
        m(ip)
        torch.cuda.nvtx.range_pop()
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        print(start.elapsed_time(end))
