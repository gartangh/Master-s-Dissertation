using Revise
using Test
using Flux
using CuArrays
using CUDAdrv

test = reshape(collect(1:7*7*256*16),(7,7,256,16)) |> gpu
shuffle_group = ShuffledGroupedConvolutions(GroupedConvolutions(+, [Conv((1,1), 256=>64, pad=(0,0)) for _ in 1:2]..., split=false),
                                       ChannelShuffle(2)) |> gpu

# function benchmark()
#     for _ in 1:1000
#         shuffle_group(input)
#     endGroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
#           #                            ChannelShuffle(2)
# end

# println("Benchmarking:")
# benchmark()
# @time benchmark()
# println("Done.")

println("Profiling:")
println(size(shuffle_group(test)))
# CUDAdrv.@profile shuffle_group(test)
println("Done.")



# # DOCUMENTATION ShuffleNet
# # ShuffleNet v1 Shufflenet unit with stride=1 stage 2 and using 2 groups
# function shufflenet_v1_stride1()
# println("shufflenet v1 stride=1")
# i  = randn(28,28,200,16)
# c  = Chain(ShuffledGroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
#           #ShuffledGroupedConvolutions(GroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
#           #                            ChannelShuffle(2)),
#            DepthwiseConv((3,3), 64=>64, pad=(1,1), stride=(1,1)),
#            GroupedConvolutions(+, [Conv((1,1), 64=>200, pad=(0,0)) for _ in 1:2]..., split=false))
# s  = SkipConnection(c, +)
# o  = s(i)
# end
#
# using Test
# println(@test size(shufflenet_v1_stride1()) == (28,28,200,16))
