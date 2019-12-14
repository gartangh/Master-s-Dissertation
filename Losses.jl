using Revise
using Flux
using Random

y = bitrand(3,16)
logŷ = randn(3,16)
ŷ =  σ.(logŷ)

println()
println(mae(ŷ, y))
println(mse(ŷ, y))
println(ce(ŷ, y))
println(lce(logŷ, y))
println(bce(ŷ, y))
println(lbce(logŷ, y))
println(fl(ŷ, y))
println(lfl(logŷ, y))
println(bfl(ŷ, y))
println(lbfl(logŷ, y))


