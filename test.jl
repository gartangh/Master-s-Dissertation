Pkg.activate(".")
using Revise
using Flux
using Torch

# split of chunks
function chunk(t::Tensor{Float32,4}, chunks::Int64)
  # n = [Ptr{Cvoid}() for _ in 1:chunks]
  dim::Int64 = 1 # reversed indexing and starting at 0
  ts = [t.ptr for _ in 1:chunks]

  Torch.atg_chunk(ts, t.ptr, chunks, dim)
  # Torch.atg_broadcast_tensors(n, ts, chunks)

  vtop::Vector{Tensor{Float32,4}} = [t for _ in 1:chunks]
  for i in 1:chunks
    vtop[i] = Tensor{Float32,4}(ts[i], Torch.on(t))
  end
  return vtop
end

# concatenate
function Base.cat(ts::Tensor{Float32,4}...; dims = 1)
  ptr = Ref(Ptr{Cvoid}())
  ts_arr = [i.ptr for i in ts]
  Torch.atg_cat(ptr, ts_arr, length(ts_arr), 4-dims)
  Tensor{Float32,4}(ptr[], Torch.on(ts[1]))
end

# grouped convolution for Tensor
function (group::GroupedConvolutions)(input::Tensor{Float32,4})
  # get input size
  w::Int64, h::Int64, c::Int64, n::Int64 = size(input)
  # number of feature maps in input
  nmaps::Int64 = c
  # number of paths of the GroupedConvolution
  npaths::Int64 = size(group.paths, 1)

  if group.split == true
    # distributes the feature maps of the input over the paths
    # throw error if number of feature maps not divisible by number of paths
    mod(nmaps, npaths) == 0 || error("the number of feature maps in the input (", nmaps, ") is not divisible by the number of paths of the GroupedConvolution (", npaths, ")")

    # number of maps per path
    nmaps_per_path::Int64 = div(nmaps, npaths)

    # calculate the output for the grouped convolutions
    # group.connection([path(input[:,:,_start_index(path_index, nmaps_per_path):_stop_index(path_index, nmaps_per_path),:]) for (path_index, path) in enumerate(group.paths)]...)
    chunks::Vector{Tensor{Float32,4}} = chunk(input, npaths)
    group.connection([path(chunks[path_index]) for (path_index, path) in enumerate(group.paths)]...)
  else
    # uses the complete input for each path
    group.connection([path(input) for (path) in group.paths]...)
  end
end

ip = rand(Float32, 10, 10, 4, 16);
c  = Conv((2,2), 4=>2);
op = c(ip);
g = GroupedConvolutions((results...) -> cat(results..., dims=3), Conv((3,3), 2=>2, pad=(1, 1), stride=(1, 1)), Conv((3,3), 2=>2, pad=(1, 1), stride=(1, 1)), split=true)
op = g(ip)

tip = tensor(ip, dev = 0);
tc  = Flux.fmap(Torch.to_tensor, c);
top = tc(tip);
tg = Flux.fmap(Torch.to_tensor, g)
top = tg(tip)
