# StaticBitVector
export StaticBitVector, StaticElementVector

"""
    StaticElementVector{N,S,C}

`N` is the length of vector, `C` is the size of storage in unit of `UInt64`, `S` is the stride defined as the `log2(# of flavors)`.
"""
struct StaticElementVector{N,S,C}
    data::NTuple{C,UInt64}
end

Base.length(::StaticElementVector{N,S,C}) where {N,S,C} = N
Base.:(==)(x::StaticElementVector, y::AbstractVector) = [x...] == [y...]
Base.:(==)(x::AbstractVector, y::StaticElementVector) = [x...] == [y...]
Base.:(==)(x::StaticElementVector{N,S,C}, y::StaticElementVector{N,S,C}) where {N,S,C} = x.data == y.data
@inline function Base.getindex(x::StaticElementVector{N,S,C}, i::Integer) where {N,S,C}
    @boundscheck i <= N || throw(BoundsError(x, i))
    i1 = (i-1)*S+1  # start point
    i2 = i*S        # stop point
    ii1 = (i1-1) ÷ 64
    ii2 = (i2-1) ÷ 64
    @inbounds if ii1 == ii2
        (x.data[ii1+1] >> (i1-ii1*64-1)) & (1<<S - 1)
    else  # cross two integers
        (x.data[ii1+1] >> (i1-ii*64-S+1)) | (x.data[ii2+1] & (1<<(i2-ii1*64) - 1))
    end
end
function StaticElementVector(nflavor::Int, x::AbstractVector)
    N = length(x)
    S = ceil(Int,log2(nflavor)) # sometimes can not devide 64.
    convert(StaticElementVector{N,S,_nints(N,S)}, x)
end
function Base.convert(::Type{StaticElementVector{N,S,C}}, x::AbstractVector) where {N,S,C}
    @assert length(x) == N
    data = zeros(UInt64,C)
    for i=1:N
        i1 = (i-1)*S+1  # start point
        i2 = i*S        # stop point
        ii1 = (i1-1) ÷ 64
        ii2 = (i2-1) ÷ 64
        @inbounds if ii1 == ii2
            data[ii1+1] |= UInt64(x[i]) << (i1-ii1*64-1)
        else  # cross two integers
            data[ii1+1] |= UInt64(x[i]) << (i1-ii1*64-1)
            data[ii2+1] |= UInt64(x[i]) >> (i2-ii1*64)
        end
    end
    return StaticElementVector{N,S,C}((data...,))
end
# joining two element sets
Base.:(|)(x::StaticElementVector{N,S,C}, y::StaticElementVector{N,S,C}) where {N,S,C} = StaticElementVector{N,S,C}(x.data .| y.data)
# intersection of two element sets
Base.:(&)(x::StaticElementVector{N,S,C}, y::StaticElementVector{N,S,C}) where {N,S,C} = StaticElementVector{N,S,C}(x.data .& y.data)
# difference of two element sets
Base.:(⊻)(x::StaticElementVector{N,S,C}, y::StaticElementVector{N,S,C}) where {N,S,C} = StaticElementVector{N,S,C}(x.data .⊻ y.data)

function onehotv(::Type{StaticElementVector{N,S,C}}, i, v) where {N,S,C}
    x = zeros(Int,N)
    x[i] = v
    return convert(StaticElementVector{N,S,C}, x)
end

##### BitVectors
const StaticBitVector{N,C} = StaticElementVector{N,1,C}
@inline function Base.getindex(x::StaticBitVector{N,C}, i::Integer) where {N,C}
    @boundscheck i <= N || throw(BoundsError(x, i))  # TODO: make this @boundscheck work.
    i -= 1
    ii = i ÷ 64
    (x.data[ii+1] >> (i-ii*64)) & 1
end

function StaticBitVector(x::AbstractVector)
    N = length(x)
    StaticBitVector{N,_nints(N,1)}((convert(BitVector, x).chunks...,))
end
function Base.convert(::Type{StaticBitVector{N,C}}, x::AbstractVector) where {N,C}
    @assert length(x) == N
    StaticBitVector(x)
end
_nints(x,s) = (x*s-1)÷64+1

@generated function Base.zero(::Type{StaticElementVector{N,S,C}}) where {N,S,C}
    Expr(:call, :(StaticElementVector{$N,$S,$C}), Expr(:tuple, zeros(UInt64, C)...))
end
staticfalses(::Type{StaticBitVector{N,C}}) where {N,C} = zero(StaticBitVector{N,C})
@generated function statictrues(::Type{StaticBitVector{N,C}}) where {N,C}
    Expr(:call, :(StaticBitVector{$N,$C}), Expr(:tuple, fill(typemax(UInt64), C)...))
end
onehotv(::Type{StaticBitVector{N,C}}, i, v) where {N,C} = v > 0 ? onehotv(StaticBitVector{N,C}, i) : zero(StaticBitVector{N,C})
function onehotv(::Type{StaticBitVector{N,C}}, i) where {N,C}
    x = falses(N)
    x[i] = true
    return StaticBitVector(x)
end
function Base.iterate(x::StaticElementVector{N,S,C}, state=1) where {N,S,C}
    if state > N
        return nothing
    else
        return x[state], state+1
    end
end

Base.show(io::IO, t::StaticElementVector) = Base.print(io, "$(join(Int.(t), ""))")