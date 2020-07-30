using Revise
using Flux
using NNlib
using CUDA
using BenchmarkTools

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

chain = Chain(
            x -> σ.(x), # caching kernel launch parameters or loading data?
            x -> σ.(x), # with cached values
            x -> σ.(x), # extra run to verify if this is equal to the previous run
            x -> relu.(x),
            x -> tanh.(x),
            x -> relu6.(x),
            x -> elu.(x),
            x -> identity(x),
            x -> leakyrelu.(x),
            x -> trelu.(x), # caching kernel launch parameters or loading data?
            x -> trelu.(x), # with cached values
            x -> trelu.(x), # extra run to verify if this is equal to the previous run
)

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
