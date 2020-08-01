using Revise
using Flux
using NNlib
using CUDA
using BenchmarkTools

DEVICE_ID = 0
println(CUDA.name(CuDevice(DEVICE_ID)))

config1 = Chain(
            BatchNorm(128),
)

config2 = Chain(
            BatchNorm(128, relu),
)

config3 = Chain(
            BatchNorm(128),
            x -> relu.(x),
)

config4 = Chain(
            x -> identity(x),
            x -> relu.(x),
)

config5 = Chain(
            x -> identity.(x),
            x -> relu.(x),
)

config6 = Chain(
            x -> identity(x),
)

config7 = Chain(
            x -> identity.(x),
)

# extend Flux function
function (c::Conv)(x::CuArray{T}) where T<:Union{Float16,Float32,Float64}
    σ, b = c.σ, reshape(c.bias, ntuple(_->1, length(c.stride))..., :, 1)
    cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
    # σ.(conv(x, c.weight, cdims) .+ b)
    conv_bias_act(x, c.weight, cdims, b, σ)
end

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

function fw5(m, ip)
    NVTX.@range "config5 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function fw6(m, ip)
    NVTX.@range "config6 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function fw7(m, ip)
    NVTX.@range "config7 CUDA.jl" begin
        CUDA.@sync m(ip)
    end
end

function main()
    m1 = config1
    m2 = config2
    m3 = config3
    m4 = config4
    m5 = config5
    m6 = config6
    m7 = config7
    ip = rand(Float32, 224, 224, 128, 16)
    GC.gc()
    CUDA.reclaim()

    gm1 = m1 |> gpu
    gm2 = m2 |> gpu
    gm3 = m3 |> gpu
    gm4 = m4 |> gpu
    gm5 = m5 |> gpu
    gm6 = m6 |> gpu
    gm7 = m7 |> gpu
    gip = ip |> gpu

    # warm-up
    fw1(gm1, gip)
    fw2(gm2, gip)
    fw3(gm3, gip)
    fw4(gm4, gip)
    fw5(gm5, gip)
    fw6(gm6, gip)
    fw7(gm7, gip)
    GC.gc()
    CUDA.reclaim()

    CUDA.@time fw1(gm1, gip)
    CUDA.@time fw2(gm2, gip)
    CUDA.@time fw3(gm3, gip)
    CUDA.@time fw4(gm4, gip)
    CUDA.@time fw5(gm5, gip)
    CUDA.@time fw6(gm6, gip)
    CUDA.@time fw7(gm7, gip)
    GC.gc()
    CUDA.reclaim()

    println()
end

main()
