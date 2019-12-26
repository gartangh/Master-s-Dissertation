using Revise
using Flux
using CuArrays
# using CUDAdrv

test = randn(Float32, (2, 3))#  |> gpu
Flux.GroupedConvolutions(Flux.mean, Flux.sum)(test)

# println("Profiling:")
# test = randn(Float32, (256, 256, 3, 1)) |> gpu
# GroupedConvolutions(test)
# test = randn(Float32, (256, 256, 3, 1)) |> gpu
# CUDAdrv.@profile GroupedConvolutions(test)
# println("DONE.")
