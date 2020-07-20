using Revise
using BenchmarkTools
using Flux, Metalhead
using CUDA
using Torch

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

function fw(m, ip)
    NVTX.@range "ResNet50 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function fw_aten(m, ip)
    NVTX.@range "ResNet50 Torch.jl" begin
        m(ip)
        Torch.sync()
    end
end

# Follow the CUDA way
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

function benchmark_cudajl(batchsize)
    m = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CUDA.reclaim()
    Torch.clear_cache()

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
    m = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CUDA.reclaim()
    Torch.clear_cache()

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
end

function benchmark_torchjl(batchsize)
    m = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CUDA.reclaim()
    Torch.clear_cache()

    tm = m.layers |> torch
    tip = tensor(ip, dev = DEVICE_ID)

    # warm-up
    fw_aten(tm, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    b = @benchmarkable(
        fw_aten($tm, $tip),
        teardown = (GC.gc(); yield(); Torch.clear_cache())
    )
    display(run(b))

    for _ in 1:5
        CUDA.@time fw_aten(tm, tip)
        GC.gc()
        yield()
        Torch.clear_cache()
    end

    println()
end
