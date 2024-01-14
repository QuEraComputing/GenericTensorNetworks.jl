import Base: mod, real, imag, reim, conj, promote_rule

export GaussMod, AbstractMod

# mod support for Gaussian integers until officially adopted into Base
mod(z::Complex{<:Integer}, n::Integer) = Complex(mod(real(z), n), mod(imag(z), n))

"""
`GaussMod{N,T}` is an alias of `Mod{N,Complex{T}}`.
It is for computing Gaussian Modulus.
"""
const GaussMod{N,T} = Mod{N,Complex{T}}
GaussMod{N}(x::T) where {N,T<:Integer} = Mod{N,Complex{T}}(x)
GaussMod{N}(x::T) where {N,T<:Complex} = Mod{N,T}(x)
Mod{N}(x::Complex{Rational{T}}) where {N,T} = Mod{N}(real(x)) + Mod{N}(imag(x)) * im
Mod{N}(re::Integer, im::Integer) where {N} = Mod{N}(Complex(re, im))

reim(x::AbstractMod) = (real(x), imag(x))
real(x::Mod{N}) where {N} = Mod{N}(real(x.val))
imag(x::Mod{N}) where {N} = Mod{N}(imag(x.val))
conj(x::Mod{N}) where {N} = Mod{N}(conj(x.val))

# ARITHMETIC
function (+)(x::GaussMod{N,T}, y::GaussMod{N,T}) where {N,T}
    xx = widen(x.val)
    yy = widen(y.val)
    zz = mod(xx + yy, N)
    return GaussMod{N,T}(zz)
end

(-)(x::GaussMod{N}) where {N} = Mod{N}(-x.val)

function (*)(x::GaussMod{N,T}, y::GaussMod{N,T}) where {N,T}
    xx = widen(x.val)
    yy = widen(y.val)
    zz = mod(xx * yy, N)
    return GaussMod{N,T}(zz)
end

function inv(x::GaussMod{N}) where {N}
    try
        a, b = reim(x)
        bot = inv(a * a + b * b)
        aa = a * bot
        bb = -b * bot
        return aa + bb * im
    catch
        error("$x is not invertible")
    end
end

is_invertible(x::GaussMod) = is_invertible(real(x * x'))
rand(::Type{GaussMod{N}}, args::Integer...) where {N} = rand(GaussMod{N,Int}, args...)
