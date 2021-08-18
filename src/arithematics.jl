export is_commutative_semiring
export Max2Poly, Polynomial, Tropical, CountingTropical, StaticBitVector, Mod, ConfigEnumerator, onehotv, ConfigSampler
export bitstringset_type, bitstringsampler_type

using Polynomials: Polynomial
using TropicalNumbers: Tropical, CountingTropical, StaticBitVector
using Mods, Primes

# patch for Tropical numbers
Base.isnan(x::Tropical) = isnan(x.n)

# pirate
Base.abs(x::Mod) = x
Base.isless(x::Mod{N}, y::Mod{N}) where N = mod(x.val, N) < mod(y.val, N)


# this function is used for testing
function is_commutative_semiring(a::T, b::T, c::T) where T
    # +
    if (a + b) + c != a + (b + c)
        @debug "(a + b) + c != a + (b + c)"
        return false
    end
    if !(a + zero(T) == zero(T) + a == a)
        @debug "!(a + zero(T) == zero(T) + a == a)"
        return false
    end
    if a + b != b + a
        @debug "a + b != b + a"
        return false
    end
    # *
    if (a * b) * c != a * (b * c)
        @debug "(a * b) * c != a * (b * c)"
        return false
    end
    if !(a * one(T) == one(T) * a == a)
        @debug "!(a * one(T) == one(T) * a == a)"
        return false
    end
    if a * b != b * a
        @debug "a * b != b * a"
        return false
    end
    # more
    if a * (b+c) != a*b + a*c
        @debug "a * (b+c) != a*b + a*c"
        return false
    end
    if (a+b) * c != a*c + b*c
        @debug "(a+b) * c != a*c + b*c"
        return false
    end
    if !(a * zero(T) == zero(T) * a == zero(T))
        @debug "!(a * zero(T) == zero(T) * a == zero(T))"
        return false
    end
    if !(a * zero(T) == zero(T) * a == zero(T))
        @debug "!(a * zero(T) == zero(T) * a == zero(T))"
        return false
    end
    return true
end

# get maximum two countings (polynomial truncated to largest two orders)
struct Max2Poly{T,TO} <: Number
    a::T
    b::T
    maxorder::TO
end

function Base.:+(a::Max2Poly, b::Max2Poly)
    if a.maxorder == b.maxorder
        return Max2Poly(a.a+b.a, a.b+b.b, a.maxorder)
    elseif a.maxorder == b.maxorder-1
        return Max2Poly(a.b+b.a, b.b, b.maxorder)
    elseif a.maxorder == b.maxorder+1
        return Max2Poly(a.a+b.b, a.b, a.maxorder)
    elseif a.maxorder < b.maxorder
        return b
    else
        return a
    end
end

function Base.:*(a::Max2Poly, b::Max2Poly)
    maxorder = a.maxorder + b.maxorder
    Max2Poly(a.a*b.b + a.b*b.a, a.b * b.b, maxorder)
end

Base.zero(::Type{Max2Poly{T,TO}}) where {T,TO} = Max2Poly(zero(T), zero(T), zero(Tropical{TO}).n)
Base.one(::Type{Max2Poly{T,TO}}) where {T,TO} = Max2Poly(zero(T), one(T), zero(TO))
Base.zero(::Max2Poly{T,TO}) where {T,TO} = zero(Max2Poly{T,TO})
Base.one(::Max2Poly{T,TO}) where {T,TO} = one(Max2Poly{T,TO})

Base.show(io::IO, x::Max2Poly) = show(io, MIME"text/plain"(), x)
function Base.show(io::IO, ::MIME"text/plain", x::Max2Poly)
    if isinf(x.maxorder)
        print(io, 0)
    else
        printpoly(io, Polynomial([x.a, x.b], :x), offset=Int(x.maxorder-1))
    end
end

struct ConfigEnumerator{N,C}
    data::Vector{StaticBitVector{N,C}}
end

Base.length(x::ConfigEnumerator{N}) where N = length(x.data)
Base.:(==)(x::ConfigEnumerator{N,C}, y::ConfigEnumerator{N,C}) where {N,C} = x.data == y.data

function Base.:+(x::ConfigEnumerator{N,C}, y::ConfigEnumerator{N,C}) where {N,C}
    length(x) == 0 && return y
    length(y) == 0 && return x
    return ConfigEnumerator{N,C}(vcat(x.data, y.data))
end

function Base.:*(x::ConfigEnumerator{L,C}, y::ConfigEnumerator{L,C}) where {L,C}
    M, N = length(x), length(y)
    M == 0 && return x
    N == 0 && return y
    z = Vector{StaticBitVector{L,C}}(undef, M*N)
    @inbounds for j=1:N, i=1:M
        z[(j-1)*M+i] = x.data[i] | y.data[j]
    end
    return ConfigEnumerator{L,C}(z)
end

Base.zero(::Type{ConfigEnumerator{N,C}}) where {N,C} = ConfigEnumerator{N,C}(StaticBitVector{N,C}[])
Base.one(::Type{ConfigEnumerator{N,C}}) where {N,C} = ConfigEnumerator{N,C}([TropicalNumbers.staticfalses(StaticBitVector{N,C})])
Base.zero(::ConfigEnumerator{N,C}) where {N,C} = zero(ConfigEnumerator{N,C})
Base.one(::ConfigEnumerator{N,C}) where {N,C} = one(ConfigEnumerator{N,C})
Base.show(io::IO, x::ConfigEnumerator) = print(io, "{", join(x.data, ", "), "}")
Base.show(io::IO, ::MIME"text/plain", x::ConfigEnumerator) = Base.show(io, x)

# the algebra sampling one of the configurations
struct ConfigSampler{N,C}
    data::StaticBitVector{N,C}
end

Base.:(==)(x::ConfigSampler{N,C}, y::ConfigSampler{N,C}) where {N,C} = x.data == y.data

function Base.:+(x::ConfigSampler{N,C}, y::ConfigSampler{N,C}) where {N,C}  # biased sampling: return `x`, maybe using random sampler is better.
    return x
end

function Base.:*(x::ConfigSampler{L,C}, y::ConfigSampler{L,C}) where {L,C}
    ConfigSampler(x.data | y.data)
end

Base.zero(::Type{ConfigSampler{N,C}}) where {N,C} = ConfigSampler{N,C}(TropicalNumbers.statictrues(StaticBitVector{N,C}))
Base.one(::Type{ConfigSampler{N,C}}) where {N,C} = ConfigSampler{N,C}(TropicalNumbers.staticfalses(StaticBitVector{N,C}))
Base.zero(::ConfigSampler{N,C}) where {N,C} = zero(ConfigSampler{N,C})
Base.one(::ConfigSampler{N,C}) where {N,C} = one(ConfigSampler{N,C})

# A patch to make `Polynomial{ConfigEnumerator}` work
function Base.:*(a::Int, y::ConfigEnumerator)
    a == 0 && return zero(y)
    a == 1 && return y
    error("multiplication between int and config enumerator is not defined.")
end
function Base.:*(a::Int, y::ConfigSampler)
    a == 0 && return zero(y)
    a == 1 && return y
    error("multiplication between int and config sampler is not defined.")
end

# convert from counting type to bitstring type
for (F,TP) in [(:bitstringset_type, :ConfigEnumerator), (:bitstringsampler_type, :ConfigSampler)]
    @eval begin
        function $F(::Type{T}, n::Int) where {OT, T<:Max2Poly{C,OT} where C}
            Max2Poly{$F(n),OT}
        end
        function $F(::Type{T}, n::Int) where {TX, T<:Polynomial{C,TX} where C}
            Polynomial{$F(n),:x}
        end
        function $F(::Type{T}, n::Int) where {TV, T<:CountingTropical{TV}}
            CountingTropical{TV, $F(n)}
        end
        function $F(n::Integer)
            C = TropicalNumbers._nints(n)
            return $TP{n, C}
        end
    end
end

# utilities for creating onehot vectors
function onehotv(::Type{Polynomial{BS,X}}, x) where {BS,X}
    Polynomial{BS,X}([zero(BS), onehotv(BS, x)])
end
function onehotv(::Type{Max2Poly{BS,OS}}, x) where {BS,OS}
    Max2Poly{BS,OS}(zero(BS), onehotv(BS, x),one(OS))
end
function onehotv(::Type{CountingTropical{TV,BS}}, x) where {TV,BS}
    CountingTropical{TV,BS}(one(TV), onehotv(BS, x))
end
onehotv(::Type{ConfigEnumerator{N,C}}, i::Integer) where {N,C} = ConfigEnumerator([TropicalNumbers.onehot(StaticBitVector{N,C}, i)])
onehotv(::Type{ConfigSampler{N,C}}, i::Integer) where {N,C} = ConfigSampler(TropicalNumbers.onehot(StaticBitVector{N,C}, i))