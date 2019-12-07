# Packages
using BenchmarkTools, Revise, Test, CUDAdrv

# Print name of GPU
println(CUDAdrv.name(CuDevice(0)))


# CPU
N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

# y .+= x
# @test all(y .== 3.0f0)

# CPU Sequential
function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)

fill!(y, 2)
print("CPU Sequential:\t")
@btime sequential_add!($y, $x)

# CPU Parallel
function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)

fill!(y, 2)
print("CPU Parallel:\t")
@btime parallel_add!($y, $x)
