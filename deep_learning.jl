using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, @epochs
using CuArrays, Revise

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images(:train)
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels(:train)
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9) |> gpu

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

dataset = (X, Y)
evalcb = () -> @show(loss(X, Y))#, accuracy(X,Y))
opt = ADAM()

@epochs 5 Flux.train!(loss, Flux.params(m), dataset, opt, cb = throttle(evalcb, 10))

println("Train accuracy: ", accuracy(X, Y))

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

println("Test accuracy: ", accuracy(tX, tY))
