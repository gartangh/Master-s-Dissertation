using Revise
using Flux
using NNlib
using CUDA
using BenchmarkTools

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))



function fw(a, b)
    NVTX.@range "Bias CUDA.jl" begin
        CUDA.@sync a .+ b
    end
end

function benchmark_cudajl(batchsize)
    a = rand(Float32, 224, 224, 128, batchsize)
    b = rand(Float32, 224, 224, 128, batchsize)
    GC.gc()
    CUDA.reclaim()

    ga = a |> gpu
    gb = b |> gpu

    # warm-up
    fw(ga, gb)
    GC.gc()
    CUDA.reclaim()

    b = @benchmarkable(
        fw($ga, $gb),
        teardown = (GC.gc(); CUDA.reclaim())
    )
    display(run(b))

    for _ in 1:5
        CUDA.@time fw(ga, gb)
        GC.gc()
        CUDA.reclaim()
    end

    println()
end

function profile_cudajl(batchsize)
    a = rand(Float32, 224, 224, 128, batchsize)
    b = rand(Float32, 224, 224, 128, batchsize)
    GC.gc()
    CUDA.reclaim()

    ga = a |> gpu
    gb = b |> gpu

    # warm-up
    fw(ga, gb)
    GC.gc()
    CUDA.reclaim()

    CUDA.@time fw(ga, gb)
    GC.gc()
    CUDA.reclaim()

    CUDA.@profile fw(ga, gb)

    println()
end
