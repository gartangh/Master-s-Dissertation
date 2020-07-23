using Revise
using Flux
using CUDA
using BenchmarkTools

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

# convolution = Conv((3, 3), 32 => 32, pad = (1, 1), stride = (1, 1)), x -> relu.(x) # relu separate
# convolution = Conv((3, 3), 128 => 128, pad = (1, 1), stride = (1, 1)) # no activation
convolution = Conv((3, 3), 128 => 128, relu, pad = (1, 1), stride = (1, 1)) # relu

# extend Flux function
function (c::Conv)(x::CuArray{T}) where T<:Union{Float16,Float32,Float64}
  σ, b = c.σ, reshape(c.bias, ntuple(_->1, length(c.stride))..., :, 1)
  cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  # σ.(conv(x, c.weight, cdims) .+ b)
  conv_bias_act(x, c.weight, cdims, b, σ)
end

function fw(m, ip)
    NVTX.@range "Conv CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    m = convolution
    ip = rand(Float32, 224, 224, 128, batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = m |> gpu
    gip = ip |> gpu

    # warm-up
    fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    b = @benchmarkable(
        fw($gm, $gip),
        teardown = (GC.gc(); CUDA.reclaim())
    )
    display(run(b))

    for _ in 1:5
        CUDA.@time fw(gm, gip)
        GC.gc()
        CUDA.reclaim()
    end

    println()
end

function profile_cudajl(batchsize)
    m = convolution
    ip = rand(Float32, 224, 224, 128, batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = m |> gpu
    gip = ip |> gpu

    # warm-up
    fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@time fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@profile fw(gm, gip)
end
