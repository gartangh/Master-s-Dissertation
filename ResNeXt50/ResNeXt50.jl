using Revise
using BenchmarkTools
using Flux
using CuArrays, CUDAdrv, CUDAnative
using Torch

DEVICE_ID = 0
println(CUDAdrv.name(CuDevice(DEVICE_ID)))

# split of chunks
function _chunk(t::Tensor{T,N}, chunks=2, dims=1) where {T,N}
  ts = [Ptr{Cvoid}() for _ in 1:chunks]
  Torch.atg_chunk(ts, t.ptr, chunks, N - dims)
  [Tensor{T,N}(ts[i], Torch.on(t)) for i in 1:chunks]
end

# concatenate
function Base.cat(ts::Tensor{Float32,N}...; dims = 1) where {N}
  ptr = Ref(Ptr{Cvoid}())
  ts_arr = [i.ptr for i in ts]
  Torch.atg_cat(ptr, ts_arr, length(ts_arr), N - dims)
  Tensor{Float32,N}(ptr[], Torch.on(ts[1]))
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
    chunks::Vector{Tensor{Float32,4}} = _chunk(input, npaths, 3)
    group.connection([path(chunks[path_index]) for (path_index, path) in enumerate(group.paths)]...)
  else
    # uses the complete input for each path
    group.connection([path(input) for (path) in group.paths]...)
  end
end

Block(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    Conv((1, 1), input_channels        => intermediate_channels, pad = (0, 0), stride = (1, 1)),
    BatchNorm(intermediate_channels, relu, ϵ = 1f-3, momentum = 0.99f0),
    GroupedConvolutions((results...) -> cat(results..., dims=3),
                        [Chain(
                            Conv((3,3), intermediate_channels÷32=>intermediate_channels÷32, pad=(1, 1), stride=(1, 1)),
                            BatchNorm(intermediate_channels÷32, relu, ϵ = 1f-3, momentum = 0.99f0),
                        ) for _ = 1:32]...,
                        split=true),
    Conv((1, 1), intermediate_channels => output_channels,       pad = (0, 0), stride = (1, 1)),
    BatchNorm(output_channels, identity, ϵ = 1f-3, momentum = 0.99f0),
)

IdentityBlock(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    SkipConnection(Block(input_channels, intermediate_channels, output_channels), +),
    x -> relu.(x),
)

ConvBlock(input_channels::Int, intermediate_channels::Int, output_channels::Int) = Chain(
    GroupedConvolutions(+ , [
        Block(input_channels, intermediate_channels, output_channels),
        Chain(
            Conv((1, 1), input_channels => output_channels, pad = (0, 0), stride = (1, 1)),
            BatchNorm(output_channels, identity, ϵ = 1f-3, momentum = 0.99f0),
            ),
    ]..., split=false),
    x -> relu.(x),
)

ResNeXt() = Chain(
    # conv1
    Conv((7, 7), 3 => 64, pad = (3, 3), stride = (2, 2)),
    BatchNorm(64, relu, ϵ = 1f-3, momentum = 0.99f0),

    # conv2
    MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
    ConvBlock(64, 128, 256),
    IdentityBlock(256, 128, 256),
    IdentityBlock(256, 128, 256),

    # conv3
    MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
    ConvBlock(256, 256, 512),
    IdentityBlock(512, 256, 512),
    IdentityBlock(512, 256, 512),
    IdentityBlock(512, 256, 512),

    # conv4
    MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
    ConvBlock(512, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    IdentityBlock(1024, 512, 1024),
    IdentityBlock(1024, 512, 1024),

    # conv5
    MaxPool((3, 3), pad = (1, 1), stride = (2, 2)),
    ConvBlock(1024, 1024, 2048),
    IdentityBlock(2048, 1024, 2048),
    IdentityBlock(2048, 1024, 2048),
    # Global Mean Pooling layer
    GlobalMeanPool(),
    flatten,
    # Fully connected layer with softmax activation
    Dense(2048, 1000),
    softmax,
)

function fw_aten(m, ip)
    NVTX.@range "Profiling Torch.jl" begin
        m(ip)
        Torch.sync()
    end
end

function fw(m, ip)
    NVTX.@range "Profiling Julia" begin
        CuArrays.@sync m(ip)
    end
end

# Follow the CuArrays way
function (tbn::BatchNorm)(x::Tensor)
  tbn.λ.(Torch.batchnorm(
    x,
    tbn.γ,
    tbn.β,
    tbn.μ,
    tbn.σ²,
    0,
    tbn.momentum,
    tbn.ϵ,
    1,
  ))
end

to_tensor(x::AbstractArray) = tensor(x, dev = DEVICE_ID)
to_tensor(x) = x

function benchmark_julia(batchsize)
    m = ResNeXt()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    gm = m |> gpu
    gip = ip |> gpu

    # warmup
    fw(gm, gip)
    GC.gc()
    CuArrays.reclaim()

    b = @benchmarkable(
        fw($gm, $gip),
        teardown = (GC.gc(); CuArrays.reclaim())
    )
    display(run(b))

    CuArrays.@time fw(gm, gip)

    println()
end

function benchmark_torchjl(batchsize)
    m = ResNeXt()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tm = Flux.fmap(to_tensor, m)
    tip = tensor(ip, dev = DEVICE_ID)

    # warmup
    fw_aten(tm, tip)
    GC.gc()
    yield()
    Torch.clear_cache()

    b = @benchmarkable(
        fw_aten($tm, $tip),
        teardown = (GC.gc(); yield(); Torch.clear_cache())
    )
    display(run(b))

    CuArrays.@time fw_aten(tm, tip)

    println()
end

function profile_julia(batchsize)
    m = ResNeXt()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    gm = m |> gpu
    gip = ip |> gpu

    # warmup
    fw(gm, gip)
    GC.gc()
    CuArrays.reclaim()

    CUDAdrv.@profile fw(gm, gip)

    println()
end

function profile_torchjl(batchsize)
    m = ResNeXt()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    yield()
    CuArrays.reclaim()
    Torch.clear_cache()

    tm = Flux.fmap(to_tensor, m)
    tip = tensor(ip, dev = DEVICE_ID)

    # warmup
    fw_aten(tm, tip)
    GC.gc()
    yield()
    Torch.clear_cache()


    CUDAdrv.@profile fw_aten(tm, tip)

    println()
end
