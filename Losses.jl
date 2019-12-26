using Revise
using Flux
using CuArrays
using Random
using CUDAdrv
using CUDAnative

y = bitrand(7,7,1024,16) |> gpu
logŷ = randn(7,7,1024,16) |> gpu
ŷ =  σ.(logŷ) |> gpu

binarycrossentropy(ŷ, y)
CUDAdrv.@profile binarycrossentropy(ŷ, y)
