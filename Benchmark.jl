# imports
using Revise
using Flux
using CuArrays
using CUDAdrv
using CUDAnative
using BenchmarkTools

# define things to benchmark in a function
function benchmark(a)
    sin.(a);
    cos.(a);
end

function profile(a)
    NVTX.mark("Started Profiling sin");
    NVTX.@range "Profiling sin" begin
        sin.(a);
    end
    NVTX.mark("Finished Profiling sin");

    NVTX.mark("Started Profiling cos");
    NVTX.@range "Profiling cos" begin
        sin.(a);
    end
    NVTX.mark("Finished Profiling cos");
end

function main(; benchmarking=true, profiling=true)
    # initialization
    a = CuArrays.rand(1024,1024,128);

    # BENCHMARKIMG
    if (benchmarking == true)
        println("BENCHMARKING");

        # warmup
        benchmark(a);
        # make sure the GPU is synchronized first
        CUDAdrv.synchronize();

        # get timing
        # '$', because external variables should be explicitly interpolated into the benchmark expression!
        t = @benchmark CuArrays.@sync(benchmark($a)); #samples=100 seconds=10;
        print(IOContext(stdout, :compact => false), t);
        # get GPU allocations
        CuArrays.@time benchmark(a) # synchronizes afterwards by itself
    end

    # PROFILING
    if (profiling == true)
        println("PROFILING");

        # warmup
        profile(a);
        # make sure the GPU is synchronized first
        CUDAdrv.synchronize();

        # profile
        CUDAdrv.@profile profile(a);
    end
end

println("")
println("")
CUDAnative.device!(0)
main(benchmarking=true, profiling=true)
