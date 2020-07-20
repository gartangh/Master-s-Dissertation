using Revise
using Flux
using CUDA
using Random
using CUDA

y = bitrand(7,7,1024,16) |> gpu
logŷ = randn(7,7,1024,16) |> gpu
ŷ =  σ.(logŷ) |> gpu

binarycrossentropy(ŷ, y)
CUDA.@profile binarycrossentropy(ŷ, y)
