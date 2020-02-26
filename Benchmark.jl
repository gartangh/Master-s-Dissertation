using Revise
using Test
using Flux
using CuArrays
using CUDAdrv
using BenchmarkTools

# time on GPU, assumes synchronized
println("ELAPSED")
a = CuArrays.rand(1024,1024,128);
sin.(a); # warmup
println(Base.@elapsed sin.(a))  # WRONG!
println(CUDAdrv.@elapsed sin.(a))

# time on GPU, will synchronize by itself
println("TIME")
Base.@time sin.(a);
CuArrays.@time sin.(a);

# More robust measurements
println("BENCHMARK")
@benchmark CuArrays.@sync sin.(a)

# time on GPU, wrapper of @benchmark, with output as @time
println("BTIME")
@btime sin.(a);

# Profile larger applications
println("PROFILE")
CUDAdrv.@profile sin.(a);

# Interactively profile larger applications
# nsys launch ~/Programs/julia-1.3.1/bin/julia
# ] activate .
# using CuArrays, CUDAdrv
# a = CuArrays.rand(1024,1024,1024);
# sin.(a);
# CUDAdrv.@profile sin.(a);
