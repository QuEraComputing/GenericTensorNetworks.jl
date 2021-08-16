export is_commutative_semiring
export Max2Poly, Polynomial, Tropical, CountingTropical, ConfigTropical, StaticBitVector, Mod, ConfigEnumerator

using Polynomials: Polynomial
using TropicalNumbers: Tropical, CountingTropical, ConfigTropical, StaticBitVector
using Mods, Primes

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
struct Max2Poly{T} <: Number
    a::T
    b::T
    maxorder::Float64
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

Base.zero(::Type{Max2Poly{T}}) where T = Max2Poly(zero(T), zero(T), -Inf)
Base.one(::Type{Max2Poly{T}}) where T = Max2Poly(zero(T), one(T), 0.0)
Base.zero(::Max2Poly{T}) where T = zero(Max2Poly{T})
Base.one(::Max2Poly{T}) where T = one(Max2Poly{T})

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

# patch

function Base.:*(a::Int, y::ConfigEnumerator)
    a == 0 && return zero(y)
    a == 1 && return y
    error("multiplication between int and config enumerator is not defined.")
end