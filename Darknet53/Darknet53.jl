using Revise
using BenchmarkTools
using Flux
using CuArrays, CUDAdrv, CUDAnative
using Torch

DEVICE_ID = 0
println(CUDAdrv.name(CuDevice(DEVICE_ID)))

Darknet() = Chain(
  # 1-2
  Conv((3, 3), 3 => 32, pad = (1, 1), stride = (1, 1)),
  BatchNorm(32, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
  Conv((3, 3), 32 => 64, pad = (1, 1), stride = (2, 2)),
  BatchNorm(64, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),

  # 3-4
  SkipConnection(
    Chain(
      Conv((1, 1), 64 => 32, pad = (0, 0), stride = (1, 1)),
      BatchNorm(32, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 32 => 64, pad = (1, 1), stride = (1, 1)),
      BatchNorm(64, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    ),
    +,
  ),
  # Residual layer

  # 5
  Conv((3, 3), 64 => 128, pad = (1, 1), stride = (2, 2)),
  BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),

  # 6-9
  SkipConnection(
    Chain(
      Conv((1, 1), 128 => 64, pad = (0, 0), stride = (1, 1)),
      BatchNorm(64, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 64 => 128, pad = (1, 1), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 128 => 64, pad = (0, 0), stride = (1, 1)),
      BatchNorm(64, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 64 => 128, pad = (1, 1), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    ),
    +,
  ),

  # 10
  Conv((3, 3), 128 => 256, pad = (1, 1), stride = (2, 2)),
  BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),

  # 11-26
  SkipConnection(
    Chain(
      Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 256 => 128, pad = (0, 0), stride = (1, 1)),
      BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 128 => 256, pad = (1, 1), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    ),
    +,
  ),
  # Residual layer

  # 27
  Conv((3, 3), 256 => 512, pad = (1, 1), stride = (2, 2)),
  BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),

  # 28-43
  SkipConnection(
    Chain(
      Conv((1, 1), 512 => 256, pad = (0, 0), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
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
      Conv((1, 1), 512 => 256, pad = (0, 0), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
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
      Conv((1, 1), 512 => 256, pad = (0, 0), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 256 => 512, pad = (1, 1), stride = (1, 1)),
      BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((1, 1), 512 => 256, pad = (0, 0), stride = (1, 1)),
      BatchNorm(256, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 256 => 512, pad = (1, 1), stride = (1, 1)),
      BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    ),
    +,
  ),
  # Residual layer

  # 44
  Conv((3, 3), 512 => 1024, pad = (1, 1), stride = (2, 2)),
  BatchNorm(1024, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),

  # 45-52
  SkipConnection(
    Chain(
      Conv((1, 1), 1024 => 512, pad = (0, 0), stride = (1, 1)),
      BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
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
      Conv((1, 1), 1024 => 512, pad = (0, 0), stride = (1, 1)),
      BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 512 => 1024, pad = (1, 1), stride = (1, 1)),
      BatchNorm(1024, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    ),
    +,
  ),
  # Residual layer

  # 53
  # Global Mean Pooling layer
  GlobalMeanPool(),
  flatten,
  # Fully connected layer with softmax activation
  Dense(1024, 1000),
  softmax,
)

function fw_aten(m, ip)
    NVTX.@range "Profiling Torch.jl" begin
        m(ip)
        Torch.sync()
    end
end

function fw(m, ip)
    NVTX.@range "Profiling Julia" begin
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

function benchmark_julia(batchsize)
    m = Darknet()
    ip = rand(Float32, 256, 256, 3, batchsize)
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
    m = Darknet()
    ip = rand(Float32, 256, 256, 3, batchsize)
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

    CuArrays.@time fw_aten(tm, tip)

    println()
end

function profile_julia(batchsize)
    m = Darknet()
    ip = rand(Float32, 256, 256, 3, batchsize)
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

    CUDAdrv.@profile fw(gm, gip)

    println()
end

function profile_torchjl(batchsize)
    m = Darknet()
    ip = rand(Float32, 256, 256, 3, batchsize)
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


    CUDAdrv.@profile fw_aten(tm, tip)

    println()
end
