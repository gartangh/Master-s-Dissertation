using Revise
using BenchmarkTools
using Flux, Metalhead
using CuArrays, CUDAdrv, CUDAnative
using Torch

DEVICE_ID = 6
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

function cuarrays(batchsize)
    resnet = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    gresnet = resnet |> gpu
    gip = ip |> gpu

    # warmup
    fw(gresnet, gip)
    GC.gc()
    CuArrays.reclaim()

    b = @benchmarkable(
        fw($gresnet, $gip),
        teardown = (GC.gc(); CuArrays.reclaim())
    )
    display(run(b))

    CuArrays.@time fw(gresnet, gip)
    GC.gc()
    CuArrays.reclaim()

    NVTX.@range "Profiling CuArrays" begin
        CUDAdrv.@profile fw(gresnet, gip)
    end
    println()
end

function torch(batchsize)
    resnet = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tresnet = Flux.fmap(to_tensor, resnet.layers)
    tip = tensor(ip, dev = DEVICE_ID)

    # warmup
    fw_aten(tresnet, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    b = @benchmarkable(
        fw_aten($tresnet, $tip),
        teardown = (GC.gc(); yield(); Torch.clear_cache())
    )
    display(run(b))

    CuArrays.@time fw(tresnet, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    NVTX.@range "Profiling Torch" begin
        CUDAdrv.@profile fw_aten(tresnet, tip)
    end
    println()
end
