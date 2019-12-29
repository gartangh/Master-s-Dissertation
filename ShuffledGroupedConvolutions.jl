using Revise
using Test
using Flux
# using CuArrays
# using CUDAdrv



# DOCUMENTATION ShuffleNet
# ShuffleNet v1 Shufflenet unit with stride=1 stage 2 and using 2 groups
function shufflenet_v1_stride1()
println("shufflenet v1 stride=1")
i  = randn(28, 28, 200, 16)
c  = Chain(ShuffledGroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
          #ShuffledGroupedConvolutions(GroupedConvolutions(+, [Conv((1,1), 200=>64, pad=(0,0)) for _ in 1:2]..., split=false),
          #                            ChannelShuffle(2)),
           DepthwiseConv((3,3), 64=>64, pad=(1,1), stride=(1,1)),
           GroupedConvolutions(+, [Conv((1,1), 64=>200, pad=(0,0)) for _ in 1:2]..., split=false))
s  = SkipConnection(c, +)
o  = s(i)
end

using Test
println(@test size(shufflenet_v1_stride1()) == (28, 28, 200, 16))
