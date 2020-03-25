# imports
using BenchmarkTools
using CuArrays
using CUDAdrv
using CUDAnative
using Flux
using Revise
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
model = Chain(
                # 1
                Conv((3, 3), 3 => 32, pad=(1, 1), stride=(1, 1)),
                BatchNorm(32, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

                # 2
                Conv((3, 3), 32 => 64, pad=(1, 1), stride=(1, 1)),
                BatchNorm(64, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

                # 3-5
                Conv((3, 3), 64 => 128, pad=(1, 1), stride=(1, 1)),
                BatchNorm(128, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((1, 1), 128 => 64, pad=(0, 0), stride=(1, 1)),
                BatchNorm(64, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((3, 3), 64 => 128, pad=(1, 1), stride=(1, 1)),
                BatchNorm(128, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

                # 6-8
                Conv((3, 3), 128 => 256, pad=(1, 1), stride=(1, 1)),
                BatchNorm(256, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((1, 1), 256 => 128, pad=(0, 0), stride=(1, 1)),
                BatchNorm(128, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((3, 3), 128 => 256, pad=(1, 1), stride=(1, 1)),
                BatchNorm(256, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

                # 9-13
                Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
                BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((1, 1), 512 => 256, pad=(0, 0), stride=(1, 1)),
                BatchNorm(256, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
                BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((1, 1), 512 => 256, pad=(0, 0), stride=(1, 1)),
                BatchNorm(256, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
                BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                MaxPool((2, 2), pad=(0, 0), stride=(2, 2)),

                # 14-18
                Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
                BatchNorm(1024, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((1, 1), 1024 => 512, pad=(0, 0), stride=(1, 1)),
                BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
                BatchNorm(1024, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((1, 1), 1024 => 512, pad=(0, 0), stride=(1, 1)),
                BatchNorm(512, leakyrelu, ϵ=1f-3, momentum=0.99f0),
                Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
                BatchNorm(1024, leakyrelu, ϵ=1f-3, momentum=0.99f0),

                # 19
                Conv((1, 1), 1024 => 1000, pad=(0, 0), stride=(1, 1)),
                # Global Mean Pooling layer
                GlobalMeanPool(),
                # Flattening layer with softmax activation
                softmax,
                flatten
              );
inputs = [
           randn(Float32, (224, 224, 3, 1)),
           randn(Float32, (224, 224, 3, 16)),
           randn(Float32, (224, 224, 3, 64)),
           randn(Float32, (224, 224, 3, 256)),
           randn(Float32, (224, 224, 3, 1024)),
           randn(Float32, (224, 224, 3, 4096)),
         ];

# set parameters
main(model, inputs, GPU, benchmarking=true, profiling=false, DEVICE_ID=0);
