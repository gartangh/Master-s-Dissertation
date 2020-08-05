using Revise
using CUDA

a = CUDA.rand(Float32, 2,3,4,5)
b = CUDA.rand(Float32, 2,1,1,5)

CUDA.CUDNN.add_bias(a,b);
println("DONE")
