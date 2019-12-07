using Revise
using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, @epochs
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using CuArrays

# Darknet-53
Darknet53() = Chain(
  # 1-2
  Conv((3, 3), 3 => 32, pad=(1, 1), stride=(1, 1)),
  BatchNorm(32, leakyrelu),
  Conv((3, 3), 32 => 64, pad=(1, 1), stride=(2, 2)),
  BatchNorm(64, leakyrelu),

  # 3-4
  SkipConnection(Chain(repeat([
    Conv((1, 1), 64 => 32, pad=(0, 0), stride=(1, 1)),
    BatchNorm(32, leakyrelu),
    Conv((3, 3), 32 => 64, pad=(1, 1), stride=(1, 1)),
    BatchNorm(64, leakyrelu)
  ], 1)...), +),
  # Residual layer

  # 5
  Conv((3, 3), 64 => 128, pad=(0, 0), stride=(2, 2)),
  BatchNorm(128, leakyrelu),

  # 6-9
  SkipConnection(Chain(repeat([
    Conv((1, 1), 128 => 64, pad=(0, 0), stride=(1, 1)),
    BatchNorm(64, leakyrelu),
    Conv((3, 3), 64 => 128, pad=(1, 1), stride=(1, 1)),
    BatchNorm(128, leakyrelu)
  ], 2)...), +),
  # Residual layer

  # 10
  Conv((3, 3), 128 => 256, pad=(0, 0), stride=(2, 2)),
  BatchNorm(256, leakyrelu),

  # 11-26
  SkipConnection(Chain(repeat([
    Conv((1, 1), 256 => 128, pad=(0, 0), stride=(1, 1)),
    BatchNorm(128, leakyrelu),
    Conv((3, 3), 128 => 256, pad=(1, 1), stride=(1, 1)),
    BatchNorm(256, leakyrelu)
  ], 8)...), +),
  # Residual layer

  # 27
  Conv((3, 3), 256 => 512, pad=(0, 0), stride=(2, 2)),
  BatchNorm(512, leakyrelu),

  # 28-43
  SkipConnection(Chain(repeat([
    Conv((1, 1), 512 => 256, pad=(0, 0), stride=(1, 1)),
    BatchNorm(256, leakyrelu),
    Conv((3, 3), 256 => 512, pad=(1, 1), stride=(1, 1)),
    BatchNorm(512, leakyrelu)
  ], 8)...), +),
  # Residual layer

  # 44
  Conv((3, 3), 512 => 1024, pad=(0, 0), stride=(2, 2)),
  BatchNorm(1024, leakyrelu),

  # 45-52
  SkipConnection(Chain(repeat([
    Conv((1, 1), 1024 => 512, pad=(0, 0), stride=(1, 1)),
    BatchNorm(512, leakyrelu),
    Conv((3, 3), 512 => 1024, pad=(1, 1), stride=(1, 1)),
    BatchNorm(1024, leakyrelu)
  ], 4)...), +),
  # Residual layer

  # Global Mean Pooling layer
  GlobalMeanPool(), Flatten(),
  # Fully connected layer
  Dense(1024, 1000), softmax) |> gpu

# Function to convert the RGB image to Float64 Arrays
getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

# Fetching the train and validation data and getting them into proper shape
X = trainimgs(CIFAR10)
imgs = [getarray(X[i].img) for i in 1:50000]
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, 100)])
valset = collect(49001:50000)
valX = cat(imgs[valset]..., dims = 4) |> gpu
valY = labels[:, valset] |> gpu

# Defining the loss and accuracy functions
m = Darknet53()
loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))

# Defining the callback and the optimizer
evalcb = throttle(() -> @show(accuracy(valX, valY)), 10)
opt = ADAM()

# Starting to train models
@show(m)
@epochs 10 Flux.train!(loss, params(m), train, opt, cb = evalcb)

# Fetch the test data from Metalhead and get it into proper shape.
# CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
test = valimgs(CIFAR10)
testimgs = [getarray(test[i].img) for i in 1:10000]
testY = onehotbatch([test[i].ground_truth.class for i in 1:10000], 1:10) |> gpu
testX = cat(testimgs..., dims = 4) |> gpu

# Print the final accuracy
@show(accuracy(testX, testY))
