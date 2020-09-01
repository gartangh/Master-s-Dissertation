using Revise
using BenchmarkTools
using CUDA
using Flux

# CPU
model = Conv((3, 3), 128 => 128, relu, pad = (1, 1))
input = rand(Float32, 224, 224, 128, 16)
# model(input) # warmup
# @benchmark model($input)

# GPU
gpu_model = model |> gpu
gpu_input = input |> gpu
gpu_model(gpu_input) # warmup
@benchmark CUDA.@sync gpu_model($gpu_input)
CUDA.@time gpu_model(gpu_input)
#
# println()
