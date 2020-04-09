import torch
import torchvision.models as models

DEVICE_ID = 0
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
print(device)


def benchmark(batchsize=64):
    m = models.inception_v3(aux_logits=False).to(device)
    ip = torch.randn(batchsize, 3, 299, 299).to(device)

    # warmup
    m(ip)

    torch.cuda.nvtx.range_push("Profiling")
    m(ip)
    torch.cuda.nvtx.range_pop()


if __name__ == '__main__':
    benchmark(4)
