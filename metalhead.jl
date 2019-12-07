using Metalhead
using Metalhead: classify

vgg = VGG19()
x = rand(Float32, 224, 224, 3, 1)
vgg(x)

vgg.layers

classify(vgg, x)
