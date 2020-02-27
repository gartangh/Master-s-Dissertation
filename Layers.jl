using Revise
using Test
using Flux
using CuArrays
using CUDAdrv
using BenchmarkTools

### CPU ###

# global pooling
x = randn(Float32, 10, 10, 3, 2)
# global max pooling
gmp = GlobalMaxPool()
@test size(gmp(x)) == (1, 1, 3, 2)
# global mean pooling
gmp = GlobalMeanPool()
@test size(gmp(x)) == (1, 1, 3, 2)

# flattening
x = randn(Float32, 10, 10, 3, 2)
@test size(flatten(x)) == (300, 2)


### GPU ###

# global pooling
x = randn(Float32, 64, 64, 128, 16) |> gpu
# global max pooling
gmp = GlobalMaxPool() |> gpu
@benchmark CuArrays.@sync gmp(x)
# global mean pooling
gmp = GlobalMeanPool() |> gpu
@benchmark CuArrays.@sync gmp(x)

# flattening
@benchmark CuArrays.@sync flatten(x)
