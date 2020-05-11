# using Revise
using BenchmarkTools
using Flux
using CuArrays, CUDAdrv, CUDAnative
using Torch

DEVICE_ID = 0
println(CUDAdrv.name(CuDevice(DEVICE_ID)))

Block(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    Conv((1, 1), input_channels        => intermediate_channels, pad = (0, 0), stride = (1, 1)),
    BatchNorm(intermediate_channels, relu, ϵ = 1f-3, momentum = 0.99f0),
    GroupwiseConv((3, 3), intermediate_channels=>intermediate_channels, relu, stride=(1, 1), pad=(1, 1), groupcount=32),
    BatchNorm(intermediate_channels, identity, ϵ = 1f-3, momentum = 0.99f0),
    Conv((1, 1), intermediate_channels => output_channels,       pad = (0, 0), stride = (1, 1)),
    BatchNorm(output_channels, identity,   ϵ = 1f-3, momentum = 0.99f0),
)

IdentityBlock(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    SkipConnection(Block(input_channels, intermediate_channels, output_channels), +),
    x -> relu.(x),
)

struct ConvBlock
  block
  chain::Chain
end

function ConvBlock(input_channels::Int, intermediate_channels::Int, output_channels::Int)
    block = Block(input_channels, intermediate_channels, output_channels)
    chain::Chain = Chain(
                            Conv((1, 1), input_channels => output_channels, pad = (0, 0), stride = (1, 1)),
                            BatchNorm(output_channels, identity, ϵ = 1f-3, momentum = 0.99f0),
                        )
   ConvBlock(block, chain)
 end

Flux.@functor ConvBlock

function (cb::ConvBlock)(input)
  cb.block(input) + cb.chain(input)
end

function Base.show(io::IO, cb::ConvBlock)
  print(io, "ConvBlock(", cb.block, ", ", cb.chain, ")")
end

ResNeXt50 = Chain(
    # conv1
    Conv((7, 7), 3 => 64, pad = (3, 3), stride = (2, 2)),
    BatchNorm(64, relu, ϵ = 1f-3, momentum = 0.99f0),
    # conv2
    MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
    ConvBlock(64, 128, 256),
    IdentityBlock(256, 128, 256),
    IdentityBlock(256, 128, 256),
    # conv3
    MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
    ConvBlock(256, 256, 512),
    IdentityBlock(512, 256, 512),
    IdentityBlock(512, 256, 512),
    IdentityBlock(512, 256, 512),
    # conv4
    MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
    ConvBlock(512, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    # conv5
    MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
    ConvBlock(1024, 1024, 2048),
    IdentityBlock(2048, 1024, 2048),
    IdentityBlock(2048, 1024, 2048),

    GlobalMeanPool(), # Global Mean Pooling layer
    flatten, # Flattening operation
    Dense(2048, 1000), # Fully Connected or Dense layer
    softmax, # Softmax activation
)

function fw_aten(m, ip)
    NVTX.@range "ResNeXt50 Torch.jl" begin
        m(ip)
        Torch.sync()
    end
end

function fw(m, ip)
    NVTX.@range "ResNeXt50 Flux" begin
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
    m = ResNeXt50
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
    m = ResNeXt50
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tm = Flux.fmap(to_tensor, m)
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
