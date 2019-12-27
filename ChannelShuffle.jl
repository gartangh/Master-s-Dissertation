using Revise
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
