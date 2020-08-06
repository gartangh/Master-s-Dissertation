# using Revise
using BenchmarkTools
using Flux
using CUDA

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

Block(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    Conv((1, 1), input_channels        => intermediate_channels,        pad = (0, 0), stride = (1, 1)),
    BatchNorm(intermediate_channels, relu, ϵ = 1f-3, momentum = 0.99f0),
    GroupwiseConv((3, 3), intermediate_channels=>intermediate_channels, pad = (1, 1), stride=(1, 1), groupcount=32),
    BatchNorm(intermediate_channels, relu, ϵ = 1f-3, momentum = 0.99f0),
    Conv((1, 1), intermediate_channels => output_channels,              pad = (0, 0), stride = (1, 1)),
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
  relu.(cb.block(input) .+ cb.chain(input))
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

function fw(m, ip)
    NVTX.@range "ResNeXt50 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = ResNeXt50 |> gpu
    gip = CUDA.rand(Float32, 224, 224, 3, batchsize)

    # warm-up
    fw(gm, gip)
    fw(gm, gip)

    b = @benchmark fw($gm, gip) setup(gip=CUDA.rand(Float32, 224, 224, 3, $batchsize))
    display(b)

    for _ in 1:5
        CUDA.@time fw(gm, gip)
    end

    println()
end

function profile_cudajl(batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = ResNeXt50 |> gpu
    gip = CUDA.rand(Float32, 224, 224, 3, batchsize)

    # warm-up
    fw(gm, gip)
    fw(gm, gip)

    CUDA.@time fw(gm, gip)
    CUDA.@profile fw(gm, gip)

    println()
end
