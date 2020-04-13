using Revise
using BenchmarkTools
using Flux, Metalhead
using CuArrays, CUDAdrv, CUDAnative

Inception() = Chain(

)

function fw(m, ip)
    NVTX.@range "Profiling Julia" begin
        CuArrays.@sync m(ip)
    end
end

m = Inception()
ip = rand(Float32, 299, 299, 3, batchsize)
GC.gc()
CuArrays.reclaim()

gm = m |> gpu
gip = ip |> gpu

# warmup
fw(gm, gip)
GC.gc()
CuArrays.reclaim()

CuArrays.@time fw(gm, gip)

println()
