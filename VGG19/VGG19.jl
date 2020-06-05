using Revise
using BenchmarkTools
using Flux, Metalhead
using CuArrays, CUDAdrv, CUDAnative
using Torch

DEVICE_ID = 0
println(CUDAdrv.name(CuDevice(DEVICE_ID)))

function fw_aten(m, ip)
    NVTX.@range "VGG19 Torch.jl" begin
        m(ip)
        Torch.sync()
    end
end

function fw(m, ip)
    NVTX.@range "VGG19 Flux" begin
        CuArrays.@sync m(ip)
    end
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

function benchmark_flux(batchsize)
    m = VGG19()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    gm = m |> gpu
    gip = ip |> gpu

    # warm-up
    fw(gm, gip)
    GC.gc()
    CuArrays.reclaim()

    b = @benchmarkable(
        fw($gm, $gip),
        teardown = (GC.gc(); CuArrays.reclaim())
    )
    display(run(b))

    for _ in 1:5
        CuArrays.@time fw(gm, gip)
        GC.gc()
        CuArrays.reclaim()
    end

    println()
end

function benchmark_torchjl(batchsize)
    m = VGG19
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tm = Flux.fmap(to_tensor, m.layers)
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
        CuArrays.@time fw_aten(tm, tip)
        GC.gc()
        yield()
        Torch.clear_cache()
    end

    println()
end
