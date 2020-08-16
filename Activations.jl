using Revise
using CUDA
using NNlib

x = 2 .* (CUDA.rand(Float32, 10) .- 0.5)
println(x)
println(σ.(x))
println()

x = CUDA.rand(Float32, 10) .- 0.5
println(x)
println(relu.(x))
println()

x = 24 .* (CUDA.rand(Float32, 10) .- 0.5)
println(x)
println(relu6.(x))
println()
