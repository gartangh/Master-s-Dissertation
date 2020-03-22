using Revise
# using Metalhead
using Flux
using Torch

println("STARTING");
# resnet = ResNet();
darknet19 = Chain(
  # 1
  Conv((3, 3), 3 => 32, pad=(1, 1), stride=(1, 1)),
  BatchNorm(32, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 2
  Conv((3, 3), 32 => 64, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 3-5
  Conv((3, 3), 64 => 128, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((1, 1), 128 => 64, pad=(0, 0), stride=(1, 1)),
  BatchNorm(64, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((3, 3), 64 => 128, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 6-8
  Conv((3, 3), 128 => 256, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((1, 1), 256 => 128, pad=(0, 0), stride=(1, 1)),
  BatchNorm(128, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((3, 3), 128 => 256, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 9-13
  Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((1, 1), 512 => 256, pad=(0, 0), stride=(1, 1)),
  BatchNorm(256, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((1, 1), 512 => 256, pad=(0, 0), stride=(1, 1)),
  BatchNorm(256, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 14-18
  Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
  BatchNorm(1024, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((1, 1), 1024 => 512, pad=(0, 0), stride=(1, 1)),
  BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
  BatchNorm(1024, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((1, 1), 1024 => 512, pad=(0, 0), stride=(1, 1)),
  BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
  Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
  BatchNorm(1024, leakyrelu, ϵ=1f-3, momentum=0.99f0),

  # 19
  Conv((1, 1), 1024 => 1000, pad=(0, 0), stride=(1, 1)),
  # Global Mean Pooling layer
  GlobalMeanPool(),
  # Flattening layer with softmax activation
  softmax,
  flatten
)

# tresnet = Flux.fmap(Torch.to_tensor, resnet.layers);
tdarknet19 = Flux.fmap(Torch.to_tensor, darknet19);

ip = rand(Float32, 224, 224, 3, 1); # An RGB Image
tip = tensor(ip, dev = 0); # 0 => GPU:0 in Torch

# tresnet(tip);
tdarknet19(tip);
println("FINISHED")
println("")
