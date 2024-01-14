real(z::GaussMod{N}) where {N} = Mod{N}(real(value(z)))
imag(z::GaussMod{N}) where {N} = Mod{N}(imag(value(z)))
reim(z::GaussMod) = (real(z), imag(z))
conj(z::GaussMod{N}) where {N} = GaussMod{N}(real(z).val, -imag(z).val)

ACZQ = Union{AbstractMod,CZQ}

function (+)(x::GaussMod{N}, y::GaussMod{N}) where {N}
    z = widen(value(x)) + widen(value(y))
    GaussMod{N}(z)
end
(+)(x::GaussMod{N}, y::T) where {N,T<:ACZQ} = x + GaussMod{N}(y)
(+)(x::T, y::GaussMod{N}) where {N,T<:ACZQ} = y + GaussMod{N}(x)

(+)(x::Mod, y::T) where {T<:Complex} = GaussMod(x) + y
(+)(x::T, y::Mod) where {T<:Complex} = x + GaussMod(y)


(-)(x::GaussMod{N}) where {N} = GaussMod{N}(-x.val)
(-)(x::GaussMod{N}, y::GaussMod{N}) where {N} = x + (-y)
(-)(x::GaussMod{N}, y::T) where {N,T<:Union{CZQ,Mod}} = x + (-y)

(-)(x::Mod, y::Union{Complex,GaussMod}) = x + (-y)
(-)(x::Union{Complex,GaussMod}, y::Mod) = x + (-y)


function *(x::GaussMod{N}, y::GaussMod{N}) where {N}
    z = widemul(x.val, y.val)         # multipy with added precision
    return GaussMod{N}(z) # return with proper type
end
(*)(x::GaussMod{N}, y::T) where {N,T<:ACZQ} = x * GaussMod{N}(y)
(*)(x::T, y::GaussMod{N}) where {N,T<:ACZQ} = y * x
# (*)(x::Mod{N}, y::Union{GaussMod,Complex}) where N = GaussMod{N}(x) * y
# (*)(x::Union{GaussMod,Complex}, y::Mod) = y * x



is_invertible(x::GaussMod) = is_invertible(real(x * x'))

function inv(x::GaussMod{N}) where {N}
    if !is_invertible(x)
        error("$x is not invertible")
    end
    a = value(real(x * x'))
    return (1 // a) * conj(x)
end


function (/)(x::GaussMod{N}, y::GaussMod{N}) where {N}
    return x * inv(y)
end

(/)(x::GaussMod{N}, y::T) where {N,T<:ACZQ} = x / GaussMod{N}(y)
(/)(x::T, y::GaussMod{N}) where {N,T<:ACZQ} = GaussMod{N}(x) / y

(//)(x::GaussMod, y::GaussMod) = x/y 
(//)(x::GaussMod, y::ACZQ) = x/y 
(//)(x::ACZQ, y::GaussMod) = x/y 


function rand(::Type{GaussMod{N}}, dims::Integer...) where {N} 
    return GaussMod{N}.(rand(Complex{Int}, dims...))
end