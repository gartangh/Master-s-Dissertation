using Revise
using BenchmarkTools
using Flux, Metalhead
using CuArrays, CUDAdrv, CUDAnative

Block(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    Conv((1, 1), input_channels        => intermediate_channels, pad = (0, 0), stride = (1, 1)),
    BatchNorm(intermediate_channels, relu, ϵ = 1f-3, momentum = 0.99f0),
    GroupedConvolutions((results...) -> cat(results..., dims=3),
                        [Chain(
                            Conv((3,3), intermediate_channels÷32=>intermediate_channels÷32, pad=(1, 1), stride=(1, 1)),
                            BatchNorm(intermediate_channels, relu, ϵ = 1f-3, momentum = 0.99f0),
                        ) for _ = 1:32]...,
                        split=true),
    Conv((1, 1), intermediate_channels => output_channels,       pad = (0, 0), stride = (1, 1)),
    BatchNorm(output_channels, identity, ϵ = 1f-3, momentum = 0.99f0),
)

IdentityBlock(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    SkipConnection(Block(input_channels, intermediate_channels, output_channels), +),
    x -> relu.(x),
)

ConvBlock(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    GroupedConvolutions(+ , [
        Block(input_channels, intermediate_channels, output_channels),
        Chain(
            Conv((1, 1), input_channels => output_channels, pad = (0, 0), stride = (1, 1)),
            BatchNorm(output_channels, identity, ϵ = 1f-3, momentum = 0.99f0),
            ),
    ]..., split=false),
    x -> relu.(x),
)

ResNeXt() = Chain(
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
    # Global Mean Pooling layer
    GlobalMeanPool(),
    flatten,
    # Fully connected layer with softmax activation
    Dense(2048, 1000),
    softmax,
)

m = ResNeXt()
ip = rand(Float32, 224, 224, 3, 64)

gm = m |> gpu
gip = ip |> gpu

# warmup
gop = gm(gip)
println(size(gop))
