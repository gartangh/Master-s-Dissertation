using Revise
using Flux
# using Flux: mean, std
using CuArrays
using CUDAdrv

# Darknet-19
model = Chain(
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

  # # 19
  # Conv((1, 1), 1024 => 1000, pad=(0, 0), stride=(1, 1)),
  # # Global Mean Pooling layer
  # GlobalMeanPool(),
  # # Flattening layer with softmax activation
  # Flatten(softmax)
  ) |> gpu

# function timing(model, n::Integer, size::NTuple{4,Integer})
#   for i = 1:n
#     test = randn(Float32, size) |> gpu
#     model(test)
#   end
# end
#
# function benchmark(model, c::Integer, ns::Array{Int64,1}, size::NTuple{4,Int64})
#   println("Benchmarking")
#   for n in ns
#     results = zeros(Float32, c)
#     println(n)
#     timing(model, n, size)
#     for i in 1:c
#       print(i, ": ")
#       t = Base.@elapsed timing(model, n, size)
#       println(t)
#       results[i] = t
#     end
#     m = mean(results)
#     s = std(results)
#     println("Mean: ", m)
#     println("std: ", s)
#     println()
#   end
#   println()
# end

# benchmark(model, 5, [1, 10, 100, 1000], (224, 224, 3, 1))

println("Profiling:")
test = randn(Float32, (224, 224, 3, 1)) |> gpu
model(test)
test = randn(Float32, (224, 224, 3, 1)) |> gpu
CUDAdrv.@profile model(test)
println("DONE.")
