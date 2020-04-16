using Revise
using BenchmarkTools

leakyrelu1(x::Real, a = oftype(x / 1, 0.01)) = max(a * x, x / one(x))

# leakyrelu2(x::Float32, a::Float32=convert(Float32, 0.01)) where {T} = max(a * x, x / one(x))

leakyrelu3(x::Real, a = oftype(x / 1, 0.01)) = max(a * x, x)

leakyrelu4(x::Array{T,N}, a = convert(T, 0.01)) where {T,N}= max.(a.*x,x)

function main()
    x::Array{Float32,4} = randn(Float32, 224,224,3,1024)
    a::Float32 = convert(Float32, 0.01)

    println("1")
    leakyrelu1.(x)
    leakyrelu1.(x, a)
    @benchmark leakyrelu1.($x)
    @benchmark leakyrelu1.($x, $a)

    # println("2")
    # leakyrelu2.(x)
    # leakyrelu2.(x, a)
    # @time leakyrelu2.(x)
    # @time leakyrelu2.(x, a)

    # println("3")
    # leakyrelu3.(x)
    # leakyrelu3.(x, a)
    # @time leakyrelu3.(x)
    # @time leakyrelu3.(x, a)

    println("4")
    leakyrelu4(x)
    leakyrelu4(x, a)
    @benchmark leakyrelu4($x)
    @benchmark leakyrelu4($x, $a)

    println()
end

main()
