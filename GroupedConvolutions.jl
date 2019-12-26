using Revise
using Flux
using CuArrays
using CUDAdrv

function create_path_sum(nmaps, npaths)
  mod(nmaps, npaths) == 0 || throw(DimensionMismatch(
    "the number of feature maps in the input (",
    nmaps,
    ") is not divisible by the number of paths of the GroupedConvolution (",
    npaths,
    ")",
  ))

  nmaps_per_path = div(nmaps, npaths)

  Chain(
    Conv((1, 1), nmaps_per_path => 4, pad = (0, 0), stride = (1, 1)),
    Conv((3, 3), 4 => 4, pad = (1, 1), stride = (1, 1)),
    Conv((1, 1), 4 => nmaps, pad = (0, 0), stride = (1, 1)),
  ) |> gpu
end

function create_path_cat(nmaps, npaths)
  mod(nmaps, npaths) == 0 || throw(DimensionMismatch(
    "the number of feature maps in the input (",
    nmaps,
    ") is not divisible by the number of paths of the GroupedConvolution (",
    npaths,
    ")",
  ))

  nmaps_per_path = div(nmaps, npaths)

  Chain(
    Conv((1, 1), nmaps_per_path => 4, pad = (0, 0), stride = (1, 1)),
    Conv((3, 3), 4 => 4, pad = (1, 1), stride = (1, 1)),
  ) |> gpu
end

# sum for 32 paths
nmaps = 256
npaths = 32

# sum
println("SUM")
group = GroupedConvolutions(+, [create_path_sum(nmaps, npaths) for _ in 1:npaths]...)
println(group)

println("Profiling:")
test = randn(Float32, (7, 7, nmaps, 16)) |> gpu
println(size(group(test)))
# test = randn(Float32, (7, 7, nmaps, 16)) |> gpu
# CUDAdrv.@profile group(test)
println("DONE.")

# concat
println("CAT")
group = GroupedConvolutions(
  (intermediate_results...) -> cat(intermediate_results..., dims = 3),
  [create_path_cat(nmaps, npaths) for _ = 1:npaths]...,
)
println(group)

println("Profiling:")
test = randn(Float32, (7, 7, nmaps, 16)) |> gpu
println(size(group(test)))
test = randn(Float32, (7, 7, nmaps, 16)) |> gpu
CUDAdrv.@profile group(test)
println("DONE.")
