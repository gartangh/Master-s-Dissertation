# imports
using Revise
using BenchmarkTools
using CuArrays
using CUDAdrv
using CUDAnative
using Flux
using Metalhead
using Torch

# benchmark
function benchmark(m, i; device=CPU)
  println("----> Benchmarking");

  # warmup
  inner_benchmark(m, i);

  if (device == CPU)
    # get timing
    # '$', because external variables should be explicitly interpolated into the benchmark expression!
    t = @benchmark inner_benchmark($m, $i);
    println(IOContext(stdout, :compact => false), t);

    # get memory allocations
    @time inner_benchmark(m, i)
  else
    # make sure the GPU is synchronized first
    CUDAdrv.synchronize();

    # get timing
    # '$', because external variables should be explicitly interpolated into the benchmark expression!
    t = @benchmark CuArrays.@sync(inner_benchmark($m, $i));
    println(IOContext(stdout, :compact => false), t);
    # get memory allocations
    # synchronizes afterwards by itself
    CuArrays.@time inner_benchmark(m, i)
  end

  println("");
end

# inner_benchmark
function inner_benchmark(m, i)
  m(i);
end

# profile
function profile(m, i)
  println("----> Profiling");

  # warmup
  inner_profile(m, i);
  # make sure the GPU is synchronized first
  CUDAdrv.synchronize();

  # profile
  CUDAdrv.@profile inner_profile(m, i);

  println("");
end

# inner profile
function inner_profile(m, i)
  NVTX.mark("Started Profiling");
  NVTX.@range "Profiling" begin
    m(i);
  end
  NVTX.mark("Finished Profiling");
end

# main
function main(model, inputs, device=CPU; benchmarking=true, profiling=false, DEVICE_ID=0)
  if (device == CPU)
    println("CPU");
    for input in inputs
      println("--> ", size(input))
      model_cpu = model |> cpu;
      input_cpu = input |> cpu;
      benchmarking == true && benchmark(model_cpu, input_cpu, device=device);
      # cannot profile on CPU
      # profiling == true && profile(model_cpu, input_cpu);
      println("");
    end
    println("");
  end

  if (device == GPU)
    println("GPU:", DEVICE_ID);
    CUDAnative.device!(DEVICE_ID);
    model_gpu = model |> gpu;
    for input in inputs
      println("--> ", size(input))
      input_gpu = input |> gpu;
      benchmarking == true && benchmark(model_gpu, input_gpu, device=device);
      profiling == true && profile(model_gpu, input_gpu);
      println("");
    end
    println("");
  end

  if (device == GPU_Torch)
    println("GPU Torch:", DEVICE_ID);
    model_gpu_torch = Flux.fmap(Torch.to_tensor, model);
    for input in inputs
      println("--> ", size(input))
      input_gpu_torch = tensor(input, dev=DEVICE_ID);
      benchmarking == true && benchmark(model_gpu_torch, input_gpu_torch, device=device);
      profiling == true && profile(model_gpu_torch, input_gpu_torch);
      println("");
    end
    println("");
  end
end

# devices
@enum Device CPU GPU GPU_Torch

# initialize model and inputs
model = VGG19()
inputs = [
  randn(Float32, (224, 224, 3, 1)),
  randn(Float32, (224, 224, 3, 16)),
  # randn(Float32, (224, 224, 3, 64)),
  # randn(Float32, (224, 224, 3, 256)),
  # randn(Float32, (224, 224, 3, 1024)),
  # randn(Float32, (224, 224, 3, 4096)),
];

# set parameters
main(model, inputs, GPU, benchmarking=true, profiling=false, DEVICE_ID=0);
