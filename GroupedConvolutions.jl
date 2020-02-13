using Revise
using Test
using Flux
using CuArrays
using CUDAdrv

# sum for 32 paths
nmaps = 256
npaths = 32

# sum
println("SUM")
function create_path_sum(nmaps, npaths)
  mod(nmaps, npaths) == 0 || throw(DimensionMismatch("the number of feature maps in the input (", nmaps, ") is not divisible by the number of paths of the GroupedConvolution (", npaths, ")",))
  nmaps_per_path = div(nmaps, npaths)

  Chain(
    Conv((1, 1), nmaps_per_path => 4, pad = (0, 0), stride = (1, 1)),
    Conv((3, 3), 4 => 4, pad = (1, 1), stride = (1, 1)),
    Conv((1, 1), 4 => nmaps, pad = (0, 0), stride = (1, 1))
  )
end
group = GroupedConvolutions(+,[create_path_sum(nmaps, npaths) for _ = 1:npaths]..., split=true) |> gpu
println(group)

println("Profiling:")
test = randn(Float32, (7, 7, nmaps, 16)) |> gpu
println(size(group(test)))
# test = randn(Float32, (7, 7, nmaps, 16)) |> gpu
# CUDAdrv.@profile group(test)
println("DONE.")

# concat
println("CAT")
function create_path_cat(nmaps, npaths)
  mod(nmaps, npaths) == 0 || throw(DimensionMismatch("the number of feature maps in the input (", nmaps, ") is not divisible by the number of paths of the GroupedConvolution (", npaths, ")",))
  nmaps_per_path = div(nmaps, npaths)

  Chain(
    Conv((1, 1), nmaps_per_path => 4, pad = (0, 0), stride = (1, 1)),
    Conv((3, 3), 4 => 4, pad = (1, 1), stride = (1, 1))
  )
end
group = GroupedConvolutions((results...) -> cat(results..., dims=3), [create_path_cat(nmaps, npaths) for _ = 1:npaths]..., split=true) |> gpu
println(group)

println("Profiling:")
test = randn(Float32, (7, 7, nmaps, 16)) |> gpu
println(size(group(test)))
# test = randn(Float32, (7, 7, nmaps, 16)) |> gpu
# CUDAdrv.@profile group(test)
println("DONE.")



# # DOCUMENTATION ResNeXt
# # NO SPLIT (A)
# function resnext_no_split()
# println("no split")
# i   = randn(7,7,256,16)
# a() = Chain(Conv((1,1), 256=>4  , pad=(0,0)),
#             Conv((3,3), 4  =>4  , pad=(1,1)),
#             Conv((1,1), 4  =>256, pad=(0,0)))
# g   = GroupedConvolutions(+, [a() for _ = 1:32]..., split=false)
# s   = SkipConnection(g, +)
# o   = s(i)
# end
#
# println(@test size(resnext_no_split()) == (7,7,256,16))
#
# # NO SPLIT EARLY CONCATENATION (B)
# function resnext_no_split_early_concatenation()
# println("no split early concatenation")
# i   = randn(7,7,256,16)
# a() = Chain(Conv((1,1), 256=>4, pad=(0,0)),
#             Conv((3,3), 4  =>4, pad=(1,1)))
# b   = Chain(GroupedConvolutions((results...) -> cat(results..., dims=3), [a() for _ = 1:32]..., split=false),
#             Conv((1,1), 128=>256, pad=(0,0)))
# s   = SkipConnection(b, +)
# o   = s(i)
# end
#
# println(@test size(resnext_no_split_early_concatenation()) == (7,7,256,16))
#
# # SPLIT (C)
# function resnext_split()
# println("split")
# i = randn(7,7,256,16)
# b = Chain(Conv((1,1), 256=>128, pad=(0,0)),
#           GroupedConvolutions((results...) -> cat(results..., dims=3), [Conv((3,3), 4=>4, pad=(1,1)) for _ = 1:32]..., split=true),
#           Conv((1,1), 128=>256, pad=(0,0)))
# s = SkipConnection(b, +)
# o = s(i)
# end
#
# println(@test size(resnext_split()) == (7,7,256,16))
#
#
#
# # DOCUMENTATION Inception
# # Inception v1 (GoogLeNet) Inception block 3a
# function inception_v1()
# println("inception v1")
# i = randn(28,28,192,16)
# a =       Conv(   (1,1), 192=>64,  pad=(0,0), relu)
# b = Chain(Conv(   (1,1), 192=>96,  pad=(0,0), relu), Conv((3,3), 96 =>128, pad=(1,1), relu))
# c = Chain(Conv(   (1,1), 192=>16,  pad=(0,0), relu), Conv((5,5), 16 =>32 , pad=(2,2), relu))
# d = Chain(MaxPool((3,3), stride=1, pad=(1,1)      ), Conv((1,1), 192=>32 , pad=(0,0), relu))
# g = GroupedConvolutions((results...) -> cat(results..., dims=3), a, b, c, d, split=false)
# o = g(i)
# end
#
# using Test
# println(@test size(inception_v1()) == (28,28,256,16))
