using Revise
using Flux
using CUDA

a = rand(Float32, 2,3,4,5) .- 0.5
b = rand(Float32, 2,3,4,5) .+ 0.5
c = rand(Float32, 2,3,4,1) .+ 0.5

ga = a |> gpu
gb = b |> gpu
gc = c |> gpu

# pointwise add
 a .+  b
ga .+ gb

# leakyrelu
leakyrelu.( a, 0.2)
leakyrelu.(ga, 0.2)

# reduce
CUDNN.cudnnReduceTensor(CUDA.CUDNN.CUDNN_REDUCE_TENSOR_ADD, ga, gc)
CUDNN.cudnnReduceTensor(CUDA.CUDNN.CUDNN_REDUCE_TENSOR_MIN, ga, gc)
CUDNN.cudnnReduceTensor(CUDA.CUDNN.CUDNN_REDUCE_TENSOR_MAX, ga, gc)
