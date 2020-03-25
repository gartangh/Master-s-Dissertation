# imports
using Revise
using Flux
using CuArrays
using CUDAdrv
using CUDAnative
using BenchmarkTools

# define what to benchmark
function benchmark(i)
    sin.(i);
    cos.(i);
end

# define what to profile
function profile(i)
    NVTX.mark("Started Profiling sin");
    NVTX.@range "Profiling sin" begin
        sin.(i);
    end
    NVTX.mark("Finished Profiling sin");

    NVTX.mark("Started Profiling cos");
    NVTX.@range "Profiling cos" begin
        cos.(i);
    end
    NVTX.mark("Finished Profiling cos");
end

# main
function main(; benchmarking=true, profiling=true)
    # initialization
    i = CuArrays.rand(1024,1024,128);

    # BENCHMARKIMG
    if (benchmarking == true)
        println("BENCHMARKING");

        # warmup
        benchmark(i);
        # make sure the GPU is synchronized first
        CUDAdrv.synchronize();

        # get timing
        # '$', because external variables should be explicitly interpolated into the benchmark expression!
        t = @benchmark CuArrays.@sync(benchmark($i)); #samples=100 seconds=10;
        print(IOContext(stdout, :compact => false), t);
        # get GPU allocations
        CuArrays.@time benchmark(i) # synchronizes afterwards by itself
    end

    # PROFILING
    if (profiling == true)
        println("PROFILING");

        # warmup
        profile(i);
        # make sure the GPU is synchronized first
        CUDAdrv.synchronize();

        # profile
        CUDAdrv.@profile profile(i);
    end
end

println("");
println("");
CUDAnative.device!(0);
main(benchmarking=true, profiling=false);
