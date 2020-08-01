using Revise
using Flux
using NNlib
using CUDA
using BenchmarkTools

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

config1 = Chain(
            Conv((3, 3), 128 => 128, pad = (1, 1), stride = (1, 1)),
)

config2 = Chain(
            Conv((3, 3), 128 => 128, pad = (1, 1), stride = (1, 1)),
            x -> relu.(x),
)

# config2 = Chain(
#             Conv((3, 3), 128 => 128, relu, pad = (1, 1), stride = (1, 1)),
# )

config3 = Chain(
            Conv((3, 3), 128 => 128, pad = (1, 1), stride = (1, 1)),
            BatchNorm(128),
            x -> relu.(x),
)

# config3 = Chain(
#             Conv((3, 3), 128 => 128, pad = (1, 1), stride = (1, 1)),
#             BatchNorm(128, relu),
# )

config4 = Chain(
            Conv((3, 3), 128 => 128, pad = (1, 1), stride = (1, 1)),
            x -> relu.(x),
            BatchNorm(128),
)

# config4 = Chain(
#             Conv((3, 3), 128 => 128, relu, pad = (1, 1), stride = (1, 1)),
#             BatchNorm(128),
# )

# extend Flux function
# function (c::Conv)(x::CuArray{T}) where T<:Union{Float16,Float32,Float64}
#     σ, b = c.σ, reshape(c.bias, ntuple(_->1, length(c.stride))..., :, 1)
#     cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
#     # σ.(conv(x, c.weight, cdims) .+ b)
#     conv_bias_act(x, c.weight, cdims, b, σ)
# end

function fw1(m, ip)
    NVTX.@range "config1 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function fw2(m, ip)
    NVTX.@range "config2 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function fw3(m, ip)
    NVTX.@range "config3 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function fw4(m, ip)
    NVTX.@range "config4 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function benchmark_cudajl(batchsize)
    m1 = config1
    m2 = config2
    m3 = config3
    m4 = config4
    ip = rand(Float32, 224, 224, 128, batchsize)
    GC.gc()
    CUDA.reclaim()

    gm1 = m1 |> gpu
    gm2 = m2 |> gpu
    gm3 = m3 |> gpu
    gm4 = m4 |> gpu
    gip = ip |> gpu

    # warm-up
    fw1(gm1, gip)
    fw2(gm2, gip)
    fw3(gm3, gip)
    fw4(gm4, gip)
    GC.gc()
    CUDA.reclaim()

    b = @benchmarkable(
        fw1($gm1, $gip),
        teardown = (GC.gc(); CUDA.reclaim())
    )
    display(run(b))

    b = @benchmarkable(
        fw2($gm2, $gip),
        teardown = (GC.gc(); CUDA.reclaim())
    )
    display(run(b))

    b = @benchmarkable(
        fw3($gm3, $gip),
        teardown = (GC.gc(); CUDA.reclaim())
    )
    display(run(b))

    b = @benchmarkable(
        fw4($gm4, $gip),
        teardown = (GC.gc(); CUDA.reclaim())
    )
    display(run(b))

    println()
end

function profile_cudajl(batchsize)
    m1 = config1
    m2 = config2
    m3 = config3
    m4 = config4
    ip = rand(Float32, 224, 224, 128, batchsize)
    GC.gc()
    CUDA.reclaim()

    gm1 = m1 |> gpu
    gm2 = m2 |> gpu
    gm3 = m3 |> gpu
    gm4 = m4 |> gpu
    gip = ip |> gpu

    # warm-up
    fw1(gm1, gip)
    fw2(gm2, gip)
    fw3(gm3, gip)
    fw4(gm4, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@time fw1(gm1, gip)
    CUDA.@time fw2(gm2, gip)
    CUDA.@time fw3(gm3, gip)
    CUDA.@time fw4(gm4, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@profile fw1(gm1, gip)
    CUDA.@profile fw2(gm2, gip)
    CUDA.@profile fw3(gm3, gip)
    CUDA.@profile fw4(gm4, gip)

    println()
end
