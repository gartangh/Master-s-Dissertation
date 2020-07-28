using Revise
using Flux
using NNlib
using CUDA


function main()
    x = randn(Float32, 112, 112, 128, 4) |> gpu
    dx = x
    w = randn(Float32, 3, 3, 128, 128) |> gpu
    dw = w
    y = randn(Float32, 112, 112, 128, 4) |> gpu
    dy = y
    cdims = DenseConvDims(x, dw, padding=1) |> gpu
    GC.gc()
    CUDA.reclaim()

    NNlib.∇conv_filter!(dw, x, dy, cdims)
    NNlib.∇conv_data!(dx, dy, w, cdims)

    println()
end

main()
