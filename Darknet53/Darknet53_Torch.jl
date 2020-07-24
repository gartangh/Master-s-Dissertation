using Revise
using BenchmarkTools
using Flux
using CUDA
using Torch

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

Darknet53 = Chain(
  # 1-2
  Conv((3, 3), 3 => 32, pad = (1, 1), stride = (1, 1)),
  BatchNorm(32, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
  Conv((3, 3), 32 => 64, pad = (1, 1), stride = (2, 2)),
  BatchNorm(64, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
  # 3-4
  # Residual block
  SkipConnection(
    Chain(
      Conv((1, 1), 64 => 32, pad = (0, 0), stride = (1, 1)),
      BatchNorm(32, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
      Conv((3, 3), 32 => 64, pad = (1, 1), stride = (1, 1)),
      BatchNorm(64, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
    ),
    +,
  ),
  # 5
  Conv((3, 3), 64 => 128, pad = (1, 1), stride = (2, 2)),
  BatchNorm(128, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
  # 6-9
  # Residual block
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
  # Residual block
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
  # 27
  Conv((3, 3), 256 => 512, pad = (1, 1), stride = (2, 2)),
  BatchNorm(512, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
  # 28-43
  # Residual block
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
  # 44
  Conv((3, 3), 512 => 1024, pad = (1, 1), stride = (2, 2)),
  BatchNorm(1024, leakyrelu, ϵ = 1f-3, momentum = 0.99f0),
  # 45-52
  # Residual block
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

  GlobalMeanPool(), # Global Mean Pooling layer
  flatten, # Flattening operation
  # 53
  Dense(1024, 1000), # Fully Connected or Dense layer
  softmax, # Softmax activation
)

function fw(m, ip)
    NVTX.@range "Darknet53 Torch.jl" begin
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
    m = Darknet53
    ip = rand(Float32, 256, 256, 3, batchsize)
    GC.gc()
    yield()
    Torch.clear_cache()

    tm = m |> torch
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
