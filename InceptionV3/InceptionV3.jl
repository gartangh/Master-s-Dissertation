using Revise
using BenchmarkTools
using Flux, Metalhead
using CuArrays, CUDAdrv, CUDAnative
using Torch

DEVICE_ID = 0
println(CUDAdrv.name(CuDevice(DEVICE_ID)))

Inception() = Chain(
    # TODO
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

function benchmark_julia(batchsize)
    m = Inception()
    ip = rand(Float32, 299, 299, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    gm = m |> gpu
    gip = ip |> gpu

    # warmup
    fw(gm, gip)
    GC.gc()
    CuArrays.reclaim()

    b = @benchmarkable(
        fw($gm, $gip),
        teardown = (GC.gc(); CuArrays.reclaim())
    )
    display(run(b))

    CuArrays.@time fw(gm, gip)

    println()
end

function benchmark_torchjl(batchsize)
    m = Inception()
    ip = rand(Float32, 299, 299, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tm = Flux.fmap(to_tensor, m)
    tip = tensor(ip, dev = DEVICE_ID)

    # warmup
    fw_aten(tm, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    b = @benchmarkable(
        fw_aten($tm, $tip),
        teardown = (GC.gc(); yield(); Torch.clear_cache())
    )
    display(run(b))

    CuArrays.@time fw(tm, tip)

    println()
end

function profile_julia(batchsize)
    m = Inception()
    ip = rand(Float32, 299, 299, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    gm = m |> gpu
    gip = ip |> gpu

    # warmup
    fw(gm, gip)
    GC.gc()
    CuArrays.reclaim()

    NVTX.@range "Profiling CuArrays" begin
        CUDAdrv.@profile fw(gm, gip)
    end
    println()
end

function profile_torchjl(batchsize)
    m = Inception()
    ip = rand(Float32, 299, 299, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tm = Flux.fmap(to_tensor, m)
    tip = tensor(ip, dev = DEVICE_ID)

    # warmup
    fw_aten(tm, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    NVTX.@range "Profiling Torch" begin
        CUDAdrv.@profile fw_aten(tm, tip)
    end
    println()
end
