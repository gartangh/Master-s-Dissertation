using BenchmarkTools
using Flux, Metalhead
using CuArrays, CUDAdrv
using Torch
function fw_aten(m, ip)
    m(ip)
    Torch.sync()
end
function fw(m, ip)
    CuArrays.@sync m(ip)
end
# Follow the CuArrays way
function (tbn::BatchNorm)(x::Tensor)
    tbn.λ.(Torch.batchnorm(x, tbn.γ,  tbn.β,  tbn.μ, tbn.σ², 0, tbn.momentum, tbn.ϵ, 1))
end
to_tensor(x::AbstractArray) = tensor(x, dev = :gpu)
to_tensor(x) = x
function benchmark(batchsize=64)
    resnet = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    Torch.clear_cache()
    CuArrays.reclaim()
    # CuArrays
    b = @benchmarkable(
        fw(gresnet, gip),
        setup=(gresnet = $resnet |> gpu;
               gip = gpu($ip)),
        teardown=(GC.gc(); CuArrays.reclaim()))
    display(run(b))
    println()
    # Torch
    b = @benchmarkable(
        fw_aten(tresnet, tip),
        setup=(tresnet = Flux.fmap(to_tensor, $resnet.layers);
               tip = tensor($ip, dev = :gpu)),
        teardown=(GC.gc(); yield(); Torch.clear_cache()))
    display(run(b))
    println()
    return
end
function main(batchsize=64)
    resnet = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    Torch.clear_cache()
    CuArrays.reclaim()
    # CuArrays
    gresnet = resnet |> gpu
    gip = gpu(ip)
    CuArrays.@time fw(gresnet, gip)
    GC.gc()
    CuArrays.reclaim()
    # Torch
    tresnet = Flux.fmap(to_tensor, resnet.layers)
    tip = tensor(ip, dev = :gpu)
    CuArrays.@time fw_aten(tresnet, tip)
    GC.gc()
    yield()
    Torch.clear_cache()
    return
end
function profile_cuarrays(batchsize=64)
    resnet = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()
    gresnet = resnet |> gpu
    gip = gpu(ip)
    CUDAdrv.@profile fw(gresnet, gip)
    return
end
function profile_torch(batchsize=64)
    resnet = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()
    tresnet = Flux.fmap(to_tensor, resnet.layers)
    tip = tensor(ip, dev = :gpu)
    CUDAdrv.@profile fw_aten(tresnet, tip)
    return
end
