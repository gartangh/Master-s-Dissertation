using Metalhead, Metalhead.Flux, Torch

resnet = ResNet()
tresnet = Flux.fmap(Torch.to_tensor, resnet.layers)

ip = rand(Float32, 224, 224, 3, 1) # An RGB Image
tip = tensor(ip, dev = 0) # 0 => GPU:0 in Torch

tresnet(tip);
