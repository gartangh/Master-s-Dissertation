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

function cuarrays(batchsize = 64)
    inception = Inception()
    ip = rand(Float32, 299, 299, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    ginception = inception |> gpu
    gip = ip |> gpu

    # warmup
    fw(ginception, gip)
    GC.gc()
    CuArrays.reclaim()

    b = @benchmarkable(
        fw($ginception, $gip),
        teardown = (GC.gc(); CuArrays.reclaim())
    )
    display(run(b))

    CuArrays.@time fw(ginception, gip)
    GC.gc()
    CuArrays.reclaim()

    NVTX.@range "Profiling CuArrays" begin
        CUDAdrv.@profile fw(ginception, gip)
    end
    println()
end

function torch(batchsize = 64)
    inception = Inception()
    ip = rand(Float32, 299, 299, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tinception = Flux.fmap(to_tensor, inception)
    tip = tensor(ip, dev = DEVICE_ID)

    # warmup
    fw_aten(tinception, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    b = @benchmarkable(
        fw_aten($tinception, $tip),
        teardown = (GC.gc(); yield(); Torch.clear_cache())
    )
    display(run(b))

    CuArrays.@time fw(tinception, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    NVTX.@range "Profiling Torch" begin
        CUDAdrv.@profile fw_aten(tinception, tip)
    end
    println()
end
