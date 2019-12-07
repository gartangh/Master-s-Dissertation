using Revise
using Flux
using CuArrays
# using Profile

# Darknet-19
Darknet19() = Chain(
  # 1
  Conv((3, 3), 3 => 32, pad=(1, 1), stride=(1, 1)),
  BatchNorm(32, leakyrelu),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 2
  Conv((3, 3), 32 => 64, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64, leakyrelu),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 3-5
  Conv((3, 3), 64 => 128, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128, leakyrelu),
  Conv((1, 1), 128 => 64, pad=(0, 0), stride=(1, 1)),
  BatchNorm(64, leakyrelu),
  Conv((3, 3), 64 => 128, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128, leakyrelu),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 6-8
  Conv((3, 3), 128 => 256, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256, leakyrelu),
  Conv((1, 1), 256 => 128, pad=(0, 0), stride=(1, 1)),
  BatchNorm(128, leakyrelu),
  Conv((3, 3), 128 => 256, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256, leakyrelu),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 9-13
  Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512, leakyrelu),
  Conv((1, 1), 512 => 256, pad=(0, 0), stride=(1, 1)),
  BatchNorm(256, leakyrelu),
  Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512, leakyrelu),
  Conv((1, 1), 512 => 256, pad=(0, 0), stride=(1, 1)),
  BatchNorm(256, leakyrelu),
  Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512, leakyrelu),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 14-18
  Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
  BatchNorm(1024, leakyrelu),
  Conv((1, 1), 1024 => 512, pad=(0, 0), stride=(1, 1)),
  BatchNorm(512, leakyrelu),
  Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
  BatchNorm(1024, leakyrelu),
  Conv((1, 1), 1024 => 512, pad=(0, 0), stride=(1, 1)),
  BatchNorm(512, leakyrelu),
  Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
  BatchNorm(1024, leakyrelu),
  MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

  # 19
  Conv((1, 1), 1024 => 1000, pad=(0, 0), stride=(1, 1)),
  # Global Mean Pooling layer
  GlobalMeanPool(),
  # Flattening layer with softmax activation
  Flatten(softmax)) |> gpu


# Darknet-53
Darknet53() = Chain(
  # 1-2
  Conv((3, 3), 3 => 32, pad=(1, 1), stride=(1, 1)),
  BatchNorm(32, leakyrelu),
  Conv((3, 3), 32 => 64, pad=(1, 1), stride=(2, 2)),
  BatchNorm(64, leakyrelu),

  # 3-4
  SkipConnection(Chain(repeat([
    Conv((1, 1), 64 => 32, pad=(0, 0), stride=(1, 1)),
    BatchNorm(32, leakyrelu),
    Conv((3, 3), 32 => 64, pad=(1, 1), stride=(1, 1)),
    BatchNorm(64, leakyrelu)
  ], 1)...), +),
  # Residual layer

  # 5
  Conv((3, 3), 64 => 128, pad=(0, 0), stride=(2, 2)),
  BatchNorm(128, leakyrelu),

  # 6-9
  SkipConnection(Chain(repeat([
    Conv((1, 1), 128 => 64, pad=(0, 0), stride=(1, 1)),
    BatchNorm(64, leakyrelu),
    Conv((3, 3), 64 => 128, pad=(1, 1), stride=(1, 1)),
    BatchNorm(128, leakyrelu)
  ], 2)...), +),
  # Residual layer

  # 10
  Conv((3, 3), 128 => 256, pad=(0, 0), stride=(2, 2)),
  BatchNorm(256, leakyrelu),

  # 11-26
  SkipConnection(Chain(repeat([
    Conv((1, 1), 256 => 128, pad=(0, 0), stride=(1, 1)),
    BatchNorm(128, leakyrelu),
    Conv((3, 3), 128 => 256, pad=(1, 1), stride=(1, 1)),
    BatchNorm(256, leakyrelu)
  ], 8)...), +),
  # Residual layer

  # 27
  Conv((3, 3), 256 => 512, pad=(0, 0), stride=(2, 2)),
  BatchNorm(512, leakyrelu),

  # 28-43
  SkipConnection(Chain(repeat([
    Conv((1, 1), 512 => 256, pad=(0, 0), stride=(1, 1)),
    BatchNorm(256, leakyrelu),
    Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
    BatchNorm(512, leakyrelu)
  ], 8)...), +),
  # Residual layer

  # 44
  Conv((3, 3), 512 => 1024, pad=(0, 0), stride=(2, 2)),
  BatchNorm(1024, leakyrelu),

  # 45-52
  SkipConnection(Chain(repeat([
    Conv((1, 1), 1024 => 512, pad=(0, 0), stride=(1, 1)),
    BatchNorm(512, leakyrelu),
    Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
    BatchNorm(1024, leakyrelu)
  ], 4)...), +),
  # Residual layer

  # Global Mean Pooling layer
  GlobalMeanPool(), Flatten(),
  # Fully connected layer
  Dense(1024, 1000), softmax) |> gpu

m = Darknet53()
@show m

function timing(n)
  for i = 1:n
    test = gpu(randn(224, 224, 3, 1))
    a = m(test)
  end
end

timing(1)
println("Timing:")
# @time timing(1)
# @time timing(10)
@time timing(100)
# @time timing(1000)

# println("Profiling:")
# @profile timing(1)
# Profile.print()
