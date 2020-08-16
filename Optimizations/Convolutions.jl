using Revise
using Flux
using CUDA
using BenchmarkTools

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

chain = Conv((3, 3), 128 => 128, pad = (1, 1), stride = (1, 1))

function fw(m, ip)
    NVTX.@range "Activation CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    m = chain
    ip = rand(Float32, 224, 224, 128, batchsize)
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
    m = chain
    ip = rand(Float32, 224, 224, 128, batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = m |> gpu
    gip = ip |> gpu

    # warm-up
    fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@time fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@profile fw(gm, gip)

    println()
end
