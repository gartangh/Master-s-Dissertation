using Revise
using Flux
using NNlib
using CUDA


function main()
    x = randn(Float32, 224, 224, 128, 4) |> gpu
    dx = similar(x)
    w = randn(Float32, 3, 3, 128, 128) |> gpu
    dw = similar(w)
    y = randn(Float32, 224, 224, 128, 4) |> gpu
    dy = similar(y)
    b = randn(Float32, 1, 1, 128, 1) |> gpu
    db = similar(b)
    GC.gc()
    CUDA.reclaim()

    # # Convolution
    cdims = DenseConvDims(x, dw, padding = 1)
    NNlib.∇conv_filter!(dw, x, dy, cdims)
    NNlib.∇conv_data!(dx, dy, w, cdims)
    GC.gc()
    CUDA.reclaim()

    # Pooling
    pdims = PoolDims(x, 2)
    NNlib.∇maxpool!(dx, dy, y, x, pdims)
    NNlib.∇meanpool!(dx, dy, y, x, pdims)
    GC.gc()
    CUDA.reclaim()

    # Bias
    CUDA.CUDNN.∇conv_bias!(db, dy) # not in NNlib
    GC.gc()
    CUDA.reclaim()

    # Softmax
    NNlib.∇softmax!(dx, dy, y)
    NNlib.∇logsoftmax!(dx, dy, y)
    GC.gc()
    CUDA.reclaim()

    println()
end

main()
