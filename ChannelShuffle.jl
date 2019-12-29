using Revise
using Test
using Flux
using CuArrays
using CUDAdrv

input = reshape(collect(1:7*7*256*16),(7,7,256,16)) |> gpu
shuffle = ChannelShuffle(8)

# function benchmark()
#     for _ in 1:1000
#         shuffle(input)
#     end
# end

# println("Benchmarking:")
# benchmark()
# @time benchmark()
# println("Done.")

println("Profiling:")
shuffle(input)
CUDAdrv.@profile shuffle(input)
println("Done.")



# # DOCUMENTATION ShuffleNet
# # ShuffleNet v1 Shufflenet unit with stride=1 stage 2 and using 2 groups
# function shufflenet_v1_stride1()
# println("shufflenet v1 stride=1")
# i  = randn(28,28,200,16)
# c  = Chain(GroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
#            ChannelShuffle(2),
#            DepthwiseConv((3,3), 64=>64, pad=(1,1), stride=(1,1)),
#            GroupedConvolutions(+, [Conv((1,1), 64=>200, pad=(0,0)) for _ in 1:2]..., split=false))
# s  = SkipConnection(c, +)
# o  = s(i)
# end
#
# using Test
# println(@test size(shufflenet_v1_stride1()) == (28,28,200,16))
#
#
#
# # ShuffleNet v1 Shufflenet unit with stride=2 stage 2 and using 2 groups
# function shufflenet_v1_stride2()
# println("shufflenet v1 stride=2")
# i  = randn(56,56,24,16)
# a  = MeanPool((3,3), pad=(1,1), stride=(2,2))
# b  = Chain(GroupedConvolutions(+, [Conv((1,1), 24=>64 , pad=(0,0)) for _ in 1:2]..., split=false),
#            ChannelShuffle(2),
#            DepthwiseConv((3,3), 64=>64, pad=(1,1), stride=(2,2)),
#            GroupedConvolutions(+, [Conv((1,1), 64=>176, pad=(0,0)) for _ in 1:2]..., split=false))
# g  = GroupedConvolutions((results...) -> cat(results..., dims=3), a, b, split=false)
# o  = g(i)
# end
#
# using Test
# println(@test size(shufflenet_v1_stride2()) == (28,28,200,16))
