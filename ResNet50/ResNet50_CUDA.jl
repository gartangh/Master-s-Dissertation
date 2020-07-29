using Revise
using BenchmarkTools
using Flux, Metalhead
using CUDA

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

# extend Flux function
function (c::Conv)(x::CuArray{T}) where T<:Union{Float16,Float32,Float64}
    σ, b = c.σ, reshape(c.bias, ntuple(_->1, length(c.stride))..., :, 1)
    cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
    # σ.(conv(x, c.weight, cdims) .+ b)
    conv_bias_act(x, c.weight, cdims, b, σ)
end

function fw(m, ip)
    NVTX.@range "ResNet50 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    m = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
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
    m = ResNet()
    ip = rand(Float32, 224, 224, 3, batchsize)
    GC.gc()
    CUDA.reclaim()

    gm = m |> gpu
    gip = ip |> gpu

    # warm-up
    CUDA.@time fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@time fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@profile fw(gm, gip)
    GC.gc()
    CUDA.reclaim()

    println()
end
