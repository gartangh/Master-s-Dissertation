using Revise
using BenchmarkTools
using Flux, Metalhead
using CuArrays, CUDAdrv, CUDAnative
using Torch

DEVICE_ID = 0
println(CUDAdrv.name(CuDevice(DEVICE_ID)))

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
    vgg = VGG19()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    gvgg = vgg |> gpu
    gip = ip |> gpu

    # warmup
    fw(gvgg, gip)
    GC.gc()
    CuArrays.reclaim()

    b = @benchmarkable(
        fw($gvgg, $gip),
        teardown = (GC.gc(); CuArrays.reclaim())
    )
    display(run(b))

    CuArrays.@time fw(gvgg, gip)
    GC.gc()
    CuArrays.reclaim()

    NVTX.@range "Profiling CuArrays" begin
        CUDAdrv.@profile fw(gvgg, gip)
    end
    println()
end

function torch(batchsize = 256)
    vgg = VGG19()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tvgg = Flux.fmap(to_tensor, vgg.layers)
    tip = tensor(ip, dev = DEVICE_ID)

    # warmup
    fw_aten(tvgg, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    b = @benchmarkable(
        fw_aten($tvgg, $tip),
        teardown = (GC.gc(); yield(); Torch.clear_cache())
    )
    display(run(b))

    CuArrays.@time fw(tvgg, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    NVTX.@range "Profiling Torch" begin
        CUDAdrv.@profile fw_aten(tvgg, tip)
    end
    println()
end
