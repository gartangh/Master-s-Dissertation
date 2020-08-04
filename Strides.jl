using Revise
using CUDA

# function f(a,b)
#     l = Int32(1)
#     m = Int32(1)
#     r = []
#     for (i,j,k) in zip(size(a),size(b),strides(b))
#         if i == j
#             l *= m
#             push!(r, l)
#             m = i
#         else
#             push!(r, l)
#         end
#     end
#     return tuple(r...)
# end
#
# a = rand(Float32, 2,3,4,5);
# b = rand(Float32, 2,1,1,5);
# r = f(a,b)
# println(r)
# println(typeof(r))
# a .+ b

# function strides_A(A::CuArray{T,N}, C::CuArray{T,N}) where {T,N}
#     l = Int32(1)
#     m = Int32(1)
#     r::Vector{Int32} = []
#     for (i,j,k) in zip(size(A),size(C),strides(A))
#         if i == j
#             l *= m
#             push!(r, l)
#             m = i
#         else
#             push!(r, l)
#         end
#     end
#     return tuple(r...)
# end

# A = CUDA.rand(Float32, 2,3,4,5);
# C = CUDA.rand(Float32, 2,1,1,5);
# s = strides_A(A,C)
# println(s)
# println(typeof(s))

A = CUDA.rand(Float32, 2,3,1,5);
C = CUDA.rand(Float32, 2,1,4,5);

println("A .+ C")
# println(A .+ C)
println(typeof(A .+ C))

println("A .+ A")
# println(A .+ A)
println(typeof(A .+ A))

println("C .+ C")
println(typeof(C .+ C))

println("C .+ A")
# println(C .+ A)
println(typeof(C .+ A))
