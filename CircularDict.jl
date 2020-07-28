using Revise
using DataStructures


struct CircularDict{K,V}
    c::CircularDeque{K}
    d::Dict{K,V}
end

CircularDict{K,V}(capacity) where {K,V} = CircularDict(CircularDeque{K}(capacity), Dict{K,V}())

Base.get(cd::CircularDict, key, default) = get(cd.d, key, default)
function Base.push!(cd::CircularDict, key, value)
    if length(cd.c) == capacity(cd.c)
        oldkey = popfirst!(cd.c)
        delete!(cd.d, oldkey)
    end
    push!(cd.c, key)
    push!(cd.d, key => value)
end

Base.show(io::IO, cd::CircularDict) = print(io, "CircularDict($(cd.c), $(cd.d))")


a = 0
circulardict = CircularDict{String, Int64}(3)
println(circulardict)
for key in ["a", "b", "c", "d", "a", "a", "e", "f"]
    global a
    algo = -1
    if algo < 0
        algo = get(circulardict, key, -1)
        if algo < 0
            # not in d
            algo = a += 1 # cudnnFindConvolutionForwardAlgorithmEx()
            push!(circulardict, key, algo)
        end
    end
    # cudnnConvolutionForward(algo)
    println(circulardict)
end
