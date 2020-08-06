using Revise
using BenchmarkTools
using Flux
using CUDA

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

Darknet19 = Chain(
    # 1
    Conv((3, 3), 3 => 32, pad = (1, 1), stride = (1, 1)),
    BatchNorm(32, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    MaxPool((2, 2), pad = (0, 0), stride = (2, 2)),
    # 2
    Conv((3, 3), 32 => 64, pad = (1, 1), stride = (1, 1)),
    BatchNorm(64, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    MaxPool((2, 2), pad = (0, 0), stride = (2, 2)),
    # 3-5
    Conv((3, 3), 64 => 128, pad = (1, 1), stride = (1, 1)),
    BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((1, 1), 128 => 64, pad = (0, 0), stride = (1, 1)),
    BatchNorm(64, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((3, 3), 64 => 128, pad = (1, 1), stride = (1, 1)),
    BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    MaxPool((2, 2), pad = (0, 0), stride = (2, 2)),
    # 6-8
    Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
    BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
    BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
    BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    MaxPool((2, 2), pad = (0, 0), stride = (2, 2)),
    # 9-13
    Conv((3, 3), 256 => 512, pad = (1, 1), stride = (1, 1)),
    BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((1, 1), 512 => 256, pad = (0, 0), stride = (1, 1)),
    BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((3, 3), 256 => 512, pad = (1, 1), stride = (1, 1)),
    BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((1, 1), 512 => 256, pad = (0, 0), stride = (1, 1)),
    BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((3, 3), 256 => 512, pad = (1, 1), stride = (1, 1)),
    BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    MaxPool((2, 2), pad = (0, 0), stride = (2, 2)),
    # 14-18
    Conv((3, 3), 512 => 1024, pad = (1, 1), stride = (1, 1)),
    BatchNorm(1024, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((1, 1), 1024 => 512, pad = (0, 0), stride = (1, 1)),
    BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((3, 3), 512 => 1024, pad = (1, 1), stride = (1, 1)),
    BatchNorm(1024, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((1, 1), 1024 => 512, pad = (0, 0), stride = (1, 1)),
    BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((3, 3), 512 => 1024, pad = (1, 1), stride = (1, 1)),
    BatchNorm(1024, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    # 19
    Conv((1, 1), 1024 => 1000, pad = (0, 0), stride = (1, 1)),

    GlobalMeanPool(), # Global Mean Pooling layer
    flatten, # Flattening operation
    softmax, # Softmax activation
)

function fw(m, ip)
    NVTX.@range "Darknet19 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = Darknet19 |> gpu
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

    gm = Darknet19 |> gpu
    gip =  CUDA.rand(Float32, 224, 224, 3, batchsize)

    # warm-up
    fw(gm, gip)
    fw(gm, gip)

    CUDA.@time fw(gm, gip)
    CUDA.@profile fw(gm, gip)

    println()
end
