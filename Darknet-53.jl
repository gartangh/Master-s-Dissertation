using Revise
using Flux
using CuArrays
# using Profile

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
  # GlobalMeanPool(), Flatten(),
  # Fully connected layer
  # Dense(1024, 1000), softmax
  ) |> gpu

# function timing(model, n::Integer, size::NTuple{4,Integer})
#   for i = 1:n
#     test = randn(Float32, size) |> gpu
#     a = model(test)
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
#       time = @elapsed timing(model, n, size)
#       println(time)
#       results[i] = time
#     end
#     m = mean(results)
#     s = std(results)
#     println("Mean: ", m)
#     println("std: ", s)
#     println()
#   end
#   println()
# end

# benchmark(Darknet53(), 5, [1, 10, 100, 1000], (256, 256, 3, 1))

println("Profiling:")
test = randn(Float32, (256, 256, 3, 1)) |> gpu
model(test)
test = randn(Float32, (256, 256, 3, 1)) |> gpu
CUDAdrv.@profile model(test)
println("DONE.")
