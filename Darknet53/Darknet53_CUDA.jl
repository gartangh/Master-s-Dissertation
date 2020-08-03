using Revise
using BenchmarkTools
using Flux
using CUDA

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
    NVTX.@range "Darknet53 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    m = Darknet53
    ip = rand(Float32, 256, 256, 3, batchsize)
    GC.gc()
    CUDA.reclaim()

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
    m = Darknet53
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    CUDA.reclaim()

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
    GC.gc()
    CUDA.reclaim()

    println()
end
