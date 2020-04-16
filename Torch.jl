using Revise
using Flux
using Torch

# split of chunks
function _chunk(t::Tensor{T,N}, chunks=2, dims=1) where {T,N}
  ts = [Ptr{Cvoid}() for _ in 1:chunks]
  Torch.atg_chunk(ts, t.ptr, chunks, N - dims)
  [Tensor{T,N}(ts[i], Torch.on(t)) for i in 1:chunks]
end

# concatenate
function Base.cat(ts::Tensor{Float32,N}...; dims = 1) where {N}
  ptr = Ref(Ptr{Cvoid}())
  ts_arr = [i.ptr for i in ts]
  Torch.atg_cat(ptr, ts_arr, length(ts_arr), N - dims)
  Tensor{Float32,N}(ptr[], Torch.on(ts[1]))
end

# softmax
function _softmax(input::Tensor{T,N}, dims = 1, dtype = Torch.options[T]) where {T,N}
  ptr = Ref(Ptr{Cvoid}())
  Torch.atg_softmax(ptr, input.ptr, N - dims, dtype)
  Tensor{T,N}(ptr[], Torch.on(input))
end

ip = rand(Float32, 3, 3, 3) # (10x10x16x2) in (WxHxCxN) order
tip = tensor(ip, dev = 0)
# chunks = _chunk(tip, 2, 3)
# top = cat(chunks..., dims=3)
sip = softmax(ip, dims=1)
tsip1 = _softmax(tip, 1)
tsip2 = _softmax(tip, 2)
tsip3 = _softmax(tip, 3)
