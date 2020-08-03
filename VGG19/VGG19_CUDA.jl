using Revise
using BenchmarkTools
using Flux, Metalhead
using CUDA

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

function fw(m, ip)
    NVTX.@range "VGG19 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    m = VGG19()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = m |> gpu
    gip = ip |> gpu

    # warm-up
    fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    b = @benchmarkable(
        fw($gm, $gip),
        teardown = (GC.gc(); CUDA.reclaim())
    )
    display(run(b))

    for _ in 1:5
        CUDA.@time fw(gm, gip)
        GC.gc()
        CUDA.reclaim()
    end

    println()
end

function profile_cudajl(batchsize)
    m = VGG19()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = m |> gpu
    gip = ip |> gpu

    # warm-up
    CUDA.@time fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@time fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@profile fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    println()
end
