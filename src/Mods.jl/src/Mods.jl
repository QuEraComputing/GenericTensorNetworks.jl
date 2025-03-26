module Mods

import Base: (==), (+), (-), (*), (inv), (/), (//), (^), hash, show
import Base: rand, conj, iszero, rtoldefault, isapprox

export Mod, modulus, value, AbstractMod
export is_invertible
export CRT

abstract type AbstractMod <: Number end

"""
`Mod{m}(v)` creates a modular number in mod `m` with value `mod(v,m)`.
"""
struct Mod{N,T} <: AbstractMod
    val::T
end

# safe constructors (slower)
function Mod{N}(x::T) where {T<:Integer,N}
    @assert N isa Integer && N > 1 "modulus must be an integer and at least 2"
    S = typeof(N)
    v = mod(x, N)
    Mod{N,S}(v)
end

function Mod{N}(x::T) where {T<:Complex{<:Integer},N}
    @assert N isa Integer && N > 1 "modulus must be an integer and at least 2"
    S = Complex{typeof(N)}
    v = mod(x, N)
    Mod{N,S}(v)
end



# type casting
Mod{N,T}(x::Mod{N,T2}) where {T,N,T2} = Mod{N,T}(T(x.val))
Mod{N,T}(x::Mod{N,T}) where {T,N} = x

show(io::IO, z::Mod{N}) where {N} = print(io, "Mod{$N}($(value(z)))")
show(io::IO, ::MIME"text/plain", z::Mod{N}) where {N} = show(io, z)

"""
`modulus(a::Mod)` returns the modulus of this `Mod` number.
```
julia> a = Mod{13}(11);

julia> modulus(a)
13
```
"""
modulus(a::Mod{N}) where {N} = N

"""
`value(a::Mod)` returns the value of this `Mod` number.
```
julia> a = Mod{13}(11);

julia> value(a)
11
```
"""
value(a::Mod{N}) where {N} = mod(a.val, N)

Base.abs(a::Mod{N,<:Real} where {N}) = abs(value(a))

function hash(x::Mod, h::UInt64 = UInt64(0))
    v = value(x)
    m = modulus(x)
    return hash(v, hash(m, h))
end

# Test for equality
iszero(x::Mod{N}) where {N} = iszero(value(x))
==(x::Mod, y::Mod) = modulus(x) == modulus(y) && value(x) == value(y)
# ==(x::Mod{N}, y::Mod{N}) where {N} = iszero(value(x - y))

# Apporximate equality
rtoldefault(::Type{Mod{N,T}}) where {N,T} = rtoldefault(T)
isapprox(x::Mod, y::Mod; kwargs...) = false
isapprox(x::Mod{N}, y::Mod{N}; kwargs...) where {N} =
    isapprox(value(x), value(y); kwargs...) || isapprox(value(y), value(x); kwargs...)

# Easy arithmetic
@inline function +(x::Mod{N,T}, y::Mod{N,T}) where {N,T}
    s, flag = Base.add_with_overflow(x.val, y.val)
    if !flag
        return Mod{N,T}(s)
    end
    t = widen(x.val) + widen(y.val)    # add with added precision
    return Mod{N,T}(mod(t, N))
end


function -(x::Mod{M,T}) where {M,T<:Signed}
    if (x.val isa BigInt || x.val == typemin(T))
        return Mod{M,T}(-mod(x.val, M))
    else
        return Mod{M,T}(-x.val)
    end
end

function -(x::Mod{M,T}) where {M,T<:Unsigned}
    return Mod{M,T}(M - value(x))
end

-(x::Mod, y::Mod) = x + (-y)

@inline function *(x::Mod{N,T}, y::Mod{N,T}) where {N,T}
    p, flag = Base.mul_with_overflow(x.val, y.val)
    if !flag
        return Mod{N,T}(p)
    else
        q = widemul(x.val, y.val)         # multipy with added precision
        return Mod{N,T}(mod(q, N)) # return with proper type
    end
end

# Division stuff
"""
`is_invertible(x::Mod)` determines if `x` is invertible.
"""
function is_invertible(x::Mod{M})::Bool where {M}
    return gcd(x.val, M) == 1
end


"""
`inv(x::Mod)` gives the multiplicative inverse of `x`.
"""
@inline function inv(x::Mod{M,T}) where {M,T}
    Mod{M,T}(_invmod(x.val, M))
end
_invmod(x::Unsigned, m::Unsigned) = invmod(x, m)
# faster version of `Base.invmod`, only works for for signed types
@inline function _invmod(x::Signed, m::Signed)
    (g, v, _) = gcdx(x, m)
    if g != 1
        error("$x (mod $m) is not invertible")
    end
    return v
end

function /(x::Mod{N,T}, y::Mod{N,T}) where {N,T}
    return x * inv(y)
end

(//)(x::Mod, y::Mod) = x / y
(//)(x::Number, y::Mod{N}) where {N} = x / y
(//)(x::Mod{N}, y::Number) where {N} = x / y

Base.promote_rule(::Type{Mod{M,T1}}, ::Type{Mod{N,T2}}) where {M,N,T1,T2<:Number} =
    error("can not promote types `Mod{$M,$T1}`` and `Mod{$N,$T2}`")
Base.promote_rule(::Type{Mod{M,T1}}, ::Type{Mod{M,T2}}) where {M,T1,T2<:Number} =
    Mod{M,promote_type(T1, T2)}
Base.promote_rule(::Type{Mod{M,T1}}, ::Type{T2}) where {M,T1,T2<:Number} =
    Mod{M,promote_type(T1, T2)}
Base.promote_rule(::Type{Mod{M,T1}}, ::Type{Rational{T2}}) where {M,T1,T2} =
    Mod{M,promote_type(T1, T2)}

# Operations with rational numbers
Mod{N}(k::Rational) where {N} = Mod{N}(numerator(k)) / Mod{N}(denominator(k))
Mod{N,T}(k::Rational{T2}) where {N,T,T2} = Mod{N,T}(numerator(k)) / Mod{N,T}(denominator(k))

# Random
rand(::Type{Mod{N}}, args::Integer...) where {N} = rand(Mod{N,Int}, args...)
rand(::Type{Mod{N,T}}) where {N,T} = Mod{N}(rand(T))
rand(::Type{Mod{N,T}}, dims::Integer...) where {N,T} = Mod{N}.(rand(T, dims...))

# Chinese remainder theorem functions
"""
    `CRT([T=BigInt, ]m1, m2,...)`

Chinese Remainder Theorem.

```
julia> CRT(Int, Mod{11}(4), Mod{14}(814))
92

julia> 92%11
4

julia> 92%14
8

julia> CRT(Mod{9223372036854775783}(9223372036854775782), Mod{9223372036854775643}(9223372036854775642))
85070591730234614113402964855534653468
```

!!! note

    `CRT` uses `BigInt` by default to prevent potential integer overflow.
    If you are confident that numbers do not overflow in your application,
    please specify an optional type parameter as the first argument.
"""
function CRT(::Type{T}, remainders, primes) where {T}
    length(remainders) == length(primes) || error("size mismatch")
    isempty(remainders) && return zero(T)
    primes = convert.(T, primes)
    M = prod(primes)
    Ms = M .÷ primes
    ti = _invmod.(Ms, primes)
    mod(sum(convert.(T, remainders) .* ti .* Ms), M)
end

function CRT(::Type{T}, rs::Mod...) where {T}
    CRT(T, value.(rs), modulus.(rs))
end
CRT(rs::Mod...) = CRT(BigInt, rs...)

include("GaussMods.jl")
include("iterate.jl")

end # end of module Mods
