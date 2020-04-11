using Revise
using BenchmarkTools
using Flux
using CuArrays, CUDAdrv, CUDAnative
using Torch

DEVICE_ID = 0
println(CUDAdrv.name(CuDevice(DEVICE_ID)))

Darknet() = Chain(
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
    # Global Mean Pooling layer
    GlobalMeanPool(),
    # Flattening layer with softmax activation
    flatten,
    softmax,
)

function fw_aten(m, ip)
    m(ip)
    Torch.sync()
end

function fw(m, ip)
    CuArrays.@sync m(ip)
end

# Follow the CuArrays way
function (tbn::BatchNorm)(x::Tensor)
    tbn.λ.(Torch.batchnorm(
        x,
        tbn.γ,
        tbn.β,
        tbn.μ,
        tbn.σ²,
        0,
        tbn.momentum,
        tbn.ϵ,
        1,
    ))
end

to_tensor(x::AbstractArray) = tensor(x, dev = DEVICE_ID)
to_tensor(x) = x

function cuarrays(batchsize = 256)
    darknet = Darknet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    gdarknet = darknet |> gpu
    gip = ip |> gpu

    # warmup
    fw(gdarknet, gip)
    GC.gc()
    CuArrays.reclaim()

    b = @benchmarkable(
        fw($gdarknet, $gip),
        teardown = (GC.gc(); CuArrays.reclaim())
    )
    display(run(b))

    CuArrays.@time fw(gdarknet, gip)
    GC.gc()
    CuArrays.reclaim()

    NVTX.@range "Profiling CuArrays" begin
        CUDAdrv.@profile fw(gdarknet, gip)
    end
    println()
end

function torch(batchsize = 256)
    darknet = Darknet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tdarknet = Flux.fmap(to_tensor, darknet)
    tip = tensor(ip, dev = DEVICE_ID)

    # warmup
    fw_aten(tdarknet, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    b = @benchmarkable(
        fw_aten($tdarknet, $tip),
        teardown = (GC.gc(); yield(); Torch.clear_cache())
    )
    display(run(b))

    CuArrays.@time fw(tdarknet, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    NVTX.@range "Profiling Torch" begin
        CUDAdrv.@profile fw_aten(tdarknet, tip)
    end
    println()
end
