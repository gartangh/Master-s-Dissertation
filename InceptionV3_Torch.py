import torch
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

inception = models.inception_v3(aux_logits=False)
inception.to(device)
ip = torch.randn(16, 3, 299, 299)
ip = ip.to(device)
# warmup
op = inception(ip)

torch.cuda.nvtx.range_push("Inception_V3")
op = inception(ip)
torch.cuda.nvtx.range_pop()
