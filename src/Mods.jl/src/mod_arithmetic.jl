@inline function (+)(x::Mod{N}, y::Mod{N}) where {N}
    t = widen(x.val) + widen(y.val)    # add with added precision
    return Mod{N}(mod(t, N))
end

(+)(x::Mod{N}, y::T) where {N,T<:QZ} = x + Mod{N}(y)
(+)(y::T, x::Mod{N}) where {N,T<:QZ} = x + y


@inline function (-)(x::Mod{M}) where {M}
    return Mod{M}(-x.val)
end

@inline (-)(x::Mod, y::Mod) = x + (-y)
(-)(x::Mod, y::T) where {T<:QZ} = x + (-y)
(-)(x::T, y::Mod) where {T<:QZ} = x + (-y)

@inline function *(x::Mod{N}, y::Mod{N}) where {N}
    q = widemul(x.val, y.val)         # multipy with added precision
    return Mod{N}(q) # return with proper type
end

(*)(x::Mod{N}, y::T) where {N,T<:QZ} = x * Mod{N}(y)
(*)(x::T, y::Mod{N}) where {N,T<:QZ} = y * x

# Division stuff
"""
`is_invertible(x::Mod)` determines if `x` is invertible.
"""
@inline function is_invertible(x::Mod{M})::Bool where {M}
    return gcd(x.val, M) == 1
end

"""
`inv(x::Mod)` gives the multiplicative inverse of `x`.
"""
@inline function inv(x::Mod{M}) where {M}
    Mod{M}(_invmod(x.val, M))
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

function (/)(x::Mod{N}, y::Mod{N}) where {N}
    return x * inv(y)
end

(/)(x::Mod{N}, y::T) where {N,T<:QZ} = x / Mod{N}(y)
(/)(x::T, y::Mod{N}) where {N,T<:QZ} = Mod{N}(x) / y

(//)(x::Mod{N}, y::Mod{N}) where {N} = x / y
(//)(x::T, y::Mod{N}) where {N,T<:QZ} = x / y
(//)(x::Mod{N}, y::T) where {N,T<:QZ} = x / y


import Base: rand
rand(::Type{Mod{N}}, dims::Integer...) where {N} = Mod{N}.(rand(Int, dims...))

