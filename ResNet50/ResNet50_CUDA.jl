using Revise
using BenchmarkTools
using Flux, Metalhead
using CUDA

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

function fw(m, ip)
    NVTX.@range "ResNet50 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = ResNet() |> gpu
    gip = CUDA.rand(Float32, 224, 224, 3, batchsize)

    # warm-up
    fw(gm, gip)
    fw(gm, gip)

    b = @benchmark fw($gm, gip) setup=(gip=CUDA.rand(Float32, 224, 224, 3, $batchsize))
    display(b)

    for _ in 1:5
        CUDA.@time fw(gm, gip)
    end

    println()
end

function profile_cudajl(batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = ResNet() |> gpu
    gip = CUDA.rand(Float32, 224, 224, 3, batchsize)

    # warm-up
    fw(gm, gip)
    fw(gm, gip)

    CUDA.@time fw(gm, gip)
    CUDA.@profile fw(gm, gip)

    println()
end
