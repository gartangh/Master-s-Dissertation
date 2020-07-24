using Revise
using BenchmarkTools
using Flux, Metalhead
using CUDA
using Torch

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

function fw(m, ip)
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

function benchmark_torchjl(batchsize)
    m = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    Torch.clear_cache()

    tm = m.layers |> torch
    tip = tensor(ip, dev = DEVICE_ID)

    # warm-up
    fw(tm, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    b = @benchmarkable(
        fw($tm, $tip),
        teardown = (GC.gc(); yield(); Torch.clear_cache())
    )
    display(run(b))

    for _ in 1:5
        CUDA.@time fw(tm, tip)
        GC.gc()
        yield()
        Torch.clear_cache()
    end

    println()
end
