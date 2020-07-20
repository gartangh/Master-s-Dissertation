# using Revise
using Test
using Flux
using CUDA

# SPLIT (C)
function resnext_split()
    println("split")
    i = randn(7, 7, 256, 16)
    b = Chain(
        Conv((1, 1), 256 => 128, pad = (0, 0)),
        GroupedConvolutions(
            (results...) -> cat(results..., dims = 3),
            [Conv((3, 3), 4 => 4, pad = (1, 1)) for _ = 1:32]...,
            split = true,
        ),
        Conv((1, 1), 128 => 256, pad = (0, 0)),
    )
    s = SkipConnection(b, +)
    o = s(i)
end
println(@test size(resnext_split()) == (7, 7, 256, 16))

# SPLIT (C)
function resnext_split()
    println("split")
    i = randn(7, 7, 256, 16)
    b = Chain(
        Conv((1, 1), 256 => 128, pad = (0, 0)),
        GroupwiseConv((3, 3), 128=>128, relu, stride = (1, 1), pad = (1, 1), dilation = (1, 1), groupcount=32),
        Conv((1, 1), 128 => 256, pad = (0, 0)),
    )
    s = SkipConnection(b, +)
    o = s(i)
end
println(@test size(resnext_split()) == (7, 7, 256, 16))
