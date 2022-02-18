using Polynomials: Polynomial
using TropicalNumbers: Tropical, CountingTropical
using Mods, Primes
using Base.Cartesian
import AbstractTrees

# pirate
Base.abs(x::Mod) = x
Base.isless(x::Mod{N}, y::Mod{N}) where N = mod(x.val, N) < mod(y.val, N)


"""
    is_commutative_semiring(a::T, b::T, c::T) where T

Check if elements `a`, `b` and `c` satisfied the commutative semiring requirements.
```math
\\begin{align*}
(a \\oplus b) \\oplus c = a \\oplus (b \\oplus c) & \\hspace{5em}\\triangleright\\text{commutative monoid \$\\oplus\$ with identity \$\\mathbb{0}\$}\\\\
a \\oplus \\mathbb{0} = \\mathbb{0} \\oplus a = a &\\\\
a \\oplus b = b \\oplus a &\\\\
&\\\\
(a \\odot b) \\odot c = a \\odot (b \\odot c)  &   \\hspace{5em}\\triangleright \\text{commutative monoid \$\\odot\$ with identity \$\\mathbb{1}\$}\\\\
a \\odot  \\mathbb{1} =  \\mathbb{1} \\odot a = a &\\\\
a \\odot b = b \\odot a &\\\\
&\\\\
a \\odot (b\\oplus c) = a\\odot b \\oplus a\\odot c  &  \\hspace{5em}\\triangleright \\text{left and right distributive}\\\\
(a\\oplus b) \\odot c = a\\odot c \\oplus b\\odot c &\\\\
&\\\\
a \\odot \\mathbb{0} = \\mathbb{0} \\odot a = \\mathbb{0}
\\end{align*}
```
"""
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

"""
    TruncatedPoly{K,T,TO} <: Number
    TruncatedPoly(coeffs::Tuple, maxorder)

Polynomial truncated to largest `K` orders. `T` is the coefficients type and `TO` is the orders type.

Example
------------------------
```jldoctest; setup=(using GraphTensorNetworks)
julia> TruncatedPoly((1,2,3), 6)
x^4 + 2*x^5 + 3*x^6

julia> TruncatedPoly((1,2,3), 6) * TruncatedPoly((5,2,1), 3)
20*x^7 + 8*x^8 + 3*x^9

julia> TruncatedPoly((1,2,3), 6) + TruncatedPoly((5,2,1), 3)
x^4 + 2*x^5 + 3*x^6
```
"""
struct TruncatedPoly{K,T,TO} <: Number
    coeffs::NTuple{K,T}
    maxorder::TO
end

"""
    Max2Poly{T,TO} = TruncatedPoly{2,T,TO}
    Max2Poly(a, b, maxorder)

A shorthand of [`TruncatedPoly`](@ref){2}.
"""
const Max2Poly{T,TO} = TruncatedPoly{2,T,TO}
Max2Poly(a, b, maxorder) = TruncatedPoly((a, b), maxorder)
Max2Poly{T,TO}(a, b, maxorder) where {T,TO} = TruncatedPoly{2,T,TO}((a, b), maxorder)

function Base.:+(a::Max2Poly, b::Max2Poly)
    aa, ab = a.coeffs
    ba, bb = b.coeffs
    if a.maxorder == b.maxorder
        return Max2Poly(aa+ba, ab+bb, a.maxorder)
    elseif a.maxorder == b.maxorder-1
        return Max2Poly(ab+ba, bb, b.maxorder)
    elseif a.maxorder == b.maxorder+1
        return Max2Poly(aa+bb, ab, a.maxorder)
    elseif a.maxorder < b.maxorder
        return b
    else
        return a
    end
end

@generated function Base.:+(a::TruncatedPoly{K}, b::TruncatedPoly{K}) where K
    quote
        if a.maxorder == b.maxorder
            return TruncatedPoly(a.coeffs .+ b.coeffs, a.maxorder)
        elseif a.maxorder > b.maxorder
            offset = a.maxorder - b.maxorder
            return TruncatedPoly((@ntuple $K i->i+offset <= $K ? a.coeffs[i] + b.coeffs[i+offset] : a.coeffs[i]), a.maxorder)
        else
            offset = b.maxorder - a.maxorder
            return TruncatedPoly((@ntuple $K i->i+offset <= $K ? b.coeffs[i] + a.coeffs[i+offset] : b.coeffs[i]), b.maxorder)
        end
    end
end

@generated function Base.:*(a::TruncatedPoly{K,T}, b::TruncatedPoly{K,T}) where {K,T}
    tupleexpr = Expr(:tuple, [K-k+1 > 1 ? Expr(:call, :+, [:(a.coeffs[$(i+k-1)]*b.coeffs[$(K-i+1)]) for i=1:K-k+1]...) : :(a.coeffs[$k]*b.coeffs[$K]) for k=1:K]...)
    quote
        maxorder = a.maxorder + b.maxorder
        TruncatedPoly($tupleexpr, maxorder)
    end
end

Base.zero(::Type{TruncatedPoly{K,T,TO}}) where {K,T,TO} = TruncatedPoly(ntuple(i->zero(T), K), zero(Tropical{TO}).n)
Base.one(::Type{TruncatedPoly{K,T,TO}}) where {K,T,TO} = TruncatedPoly(ntuple(i->i==K ? one(T) : zero(T), K), zero(TO))
Base.zero(::TruncatedPoly{K,T,TO}) where {K,T,TO} = zero(TruncatedPoly{K,T,TO})
Base.one(::TruncatedPoly{K,T,TO}) where {K,T,TO} = one(TruncatedPoly{K,T,TO})

Base.show(io::IO, x::TruncatedPoly) = show(io, MIME"text/plain"(), x)
function Base.show(io::IO, ::MIME"text/plain", x::TruncatedPoly{K}) where K
    if isinf(x.maxorder)
        print(io, 0)
    else
        printpoly(io, Polynomial([x.coeffs...], :x), offset=Int(x.maxorder-K+1))
    end
end

"""
    ConfigEnumerator{N,S,C}

Set algebra for enumerating configurations, where `N` is the length of configurations,
`C` is the size of storage in unit of `UInt64`,
`S` is the bit width to store a single element in a configuration, i.e. `log2(# of flavors)`, for bitstrings, it is `1``.

Example
----------------------
```jldoctest; setup=:(using GraphTensorNetworks)
julia> a = ConfigEnumerator([StaticBitVector([1,1,1,0,0]), StaticBitVector([1,0,0,0,1])])
{11100, 10001}

julia> b = ConfigEnumerator([StaticBitVector([0,0,0,0,0]), StaticBitVector([1,0,1,0,1])])
{00000, 10101}

julia> a + b
{11100, 10001, 00000, 10101}

julia> one(a)
{00000}

julia> zero(a)
{}
```
"""
struct ConfigEnumerator{N,S,C}
    data::Vector{StaticElementVector{N,S,C}}
end

Base.length(x::ConfigEnumerator{N}) where N = length(x.data)
Base.iterate(x::ConfigEnumerator{N}) where N = iterate(x.data)
Base.iterate(x::ConfigEnumerator{N}, state) where N = iterate(x.data, state)
Base.getindex(x::ConfigEnumerator, i) = x.data[i]
Base.:(==)(x::ConfigEnumerator{N,S,C}, y::ConfigEnumerator{N,S,C}) where {N,S,C} = Set(x.data) == Set(y.data)

function Base.:+(x::ConfigEnumerator{N,S,C}, y::ConfigEnumerator{N,S,C}) where {N,S,C}
    length(x) == 0 && return y
    length(y) == 0 && return x
    return ConfigEnumerator{N,S,C}(vcat(x.data, y.data))
end

function Base.:*(x::ConfigEnumerator{L,S,C}, y::ConfigEnumerator{L,S,C}) where {L,S,C}
    M, N = length(x), length(y)
    M == 0 && return x
    N == 0 && return y
    z = Vector{StaticElementVector{L,S,C}}(undef, M*N)
    @inbounds for j=1:N, i=1:M
        z[(j-1)*M+i] = x.data[i] | y.data[j]
    end
    return ConfigEnumerator{L,S,C}(z)
end

Base.zero(::Type{ConfigEnumerator{N,S,C}}) where {N,S,C} = ConfigEnumerator{N,S,C}(StaticElementVector{N,S,C}[])
Base.one(::Type{ConfigEnumerator{N,S,C}}) where {N,S,C} = ConfigEnumerator{N,S,C}([zero(StaticElementVector{N,S,C})])
Base.zero(::ConfigEnumerator{N,S,C}) where {N,S,C} = zero(ConfigEnumerator{N,S,C})
Base.one(::ConfigEnumerator{N,S,C}) where {N,S,C} = one(ConfigEnumerator{N,S,C})
Base.show(io::IO, x::ConfigEnumerator) = print(io, "{", join(x.data, ", "), "}")
Base.show(io::IO, ::MIME"text/plain", x::ConfigEnumerator) = Base.show(io, x)

# the algebra sampling one of the configurations
"""
    ConfigSampler{N,S,C}
    ConfigSampler(elements::StaticElementVector)

The algebra for sampling one configuration, where `N` is the length of configurations,
`C` is the size of storage in unit of `UInt64`,
`S` is the bit width to store a single element in a configuration, i.e. `log2(# of flavors)`, for bitstrings, it is `1``.

!!! note
    `ConfigSampler` is a **probabilistic** commutative semiring, adding two config samplers do not give you deterministic results.

Example
----------------------
```jldoctest; setup=:(using GraphTensorNetworks, Random; Random.seed!(2))
julia> ConfigSampler(StaticBitVector([1,1,1,0,0]))
ConfigSampler{5, 1, 1}(11100)

julia> ConfigSampler(StaticBitVector([1,1,1,0,0])) + ConfigSampler(StaticBitVector([1,0,1,0,0]))
ConfigSampler{5, 1, 1}(10100)

julia> ConfigSampler(StaticBitVector([1,1,1,0,0])) * ConfigSampler(StaticBitVector([0,0,0,0,1]))
ConfigSampler{5, 1, 1}(11101)

julia> one(ConfigSampler{5, 1, 1})
ConfigSampler{5, 1, 1}(00000)

julia> zero(ConfigSampler{5, 1, 1})
ConfigSampler{5, 1, 1}(11111)
```
"""
struct ConfigSampler{N,S,C}
    data::StaticElementVector{N,S,C}
end

Base.:(==)(x::ConfigSampler{N,S,C}, y::ConfigSampler{N,S,C}) where {N,S,C} = x.data == y.data

function Base.:+(x::ConfigSampler{N,S,C}, y::ConfigSampler{N,S,C}) where {N,S,C}  # biased sampling: return `x`, maybe using random sampler is better.
    return randn() > 0.5 ? x : y
end

function Base.:*(x::ConfigSampler{L,S,C}, y::ConfigSampler{L,S,C}) where {L,S,C}
    ConfigSampler(x.data | y.data)
end

@generated function Base.zero(::Type{ConfigSampler{N,S,C}}) where {N,S,C}
    ex = Expr(:call, :(StaticElementVector{$N,$S,$C}), Expr(:tuple, fill(typemax(UInt64), C)...))
    :(ConfigSampler{N,S,C}($ex))
end
Base.one(::Type{ConfigSampler{N,S,C}}) where {N,S,C} = ConfigSampler{N,S,C}(zero(StaticElementVector{N,S,C}))
Base.zero(::ConfigSampler{N,S,C}) where {N,S,C} = zero(ConfigSampler{N,S,C})
Base.one(::ConfigSampler{N,S,C}) where {N,S,C} = one(ConfigSampler{N,S,C})

# tree config enumerator
export TreeConfigEnumerator
struct TreeConfigEnumerator{N,S,C}
    siblings::Vector{TreeConfigEnumerator{N,S,C}}
    isleaf::Bool
    data::StaticElementVector{N,S,C}
    TreeConfigEnumerator(v::Vector{TreeConfigEnumerator{N,S,C}}) where {N,S,C} = new{N,S,C}(v, false)
    TreeConfigEnumerator(data::StaticElementVector{N,S,C}) where {N,S,C} = new{N,S,C}(StaticElementVector{N,S,C}[], true, data)
end

isleaf(t::TreeConfigEnumerator) = t.isleaf
AbstractTrees.children(t::TreeConfigEnumerator) = t.siblings
AbstractTrees.printnode(io::IO, t::TreeConfigEnumerator) = isleaf(t) ? print(io, t.data) : print(io, "○")

Base.length(x::TreeConfigEnumerator{N}) where N = isleaf(x) ? 1 : prod(length, x.siblings)
Base.eltype(::Type{<:AbstractTrees.TreeIterator{TreeConfigEnumerator{N,S,C}}}) where {N,S,C} = StaticElementVector{N,S,C}
Base.IteratorEltype(::Type{<:AbstractTrees.TreeIterator{TreeConfigEnumerator{N, S, C}}}) where {N,S,C} = Base.HasEltype()a
function Base.:(==)(x::TreeConfigEnumerator{N,S,C}, y::TreeConfigEnumerator{N,S,C}) where {N,S,C}
    if isleaf(x) && isleaf(y)
        return x.data == y.data
    elseif !isleaf(x) && !isleaf(y) && length(x.siblings) == length(y.siblings)
        return all(i->x.siblings[i] == y.siblings[i], lengh(x.siblings))
    else
        return false
    end
end

function Base.:+(x::TreeConfigEnumerator{N,S,C}, y::TreeConfigEnumerator{N,S,C}) where {N,S,C}
    iszero(x) && return y
    iszero(y) && return x
    return TreeConfigEnumerator(vcat(x.siblings, y.siblings))
end

function Base.:*(x::TreeConfigEnumerator{L,S,C}, y::TreeConfigEnumerator{L,S,C}) where {L,S,C}
    iszero(x) && return x
    iszero(y) && return y
    return TreeConfigEnumerator([x, y])
end

Base.zero(::Type{TreeConfigEnumerator{N,S,C}}) where {N,S,C} = TreeConfigEnumerator(TreeConfigEnumerator{N,S,C}[])
Base.one(::Type{TreeConfigEnumerator{N,S,C}}) where {N,S,C} = TreeConfigEnumerator([TreeConfigEnumerator(zero(StaticElementVector{N,S,C}))])
Base.zero(::TreeConfigEnumerator{N,S,C}) where {N,S,C} = zero(TreeConfigEnumerator{N,S,C})
Base.one(::TreeConfigEnumerator{N,S,C}) where {N,S,C} = one(TreeConfigEnumerator{N,S,C})
# todo, check siblings too?
Base.iszero(t::TreeConfigEnumerator) = isempty(t.siblings)
Base.iterate(t::TreeConfigEnumerator) = iterate(AbstractTrees.PreOrderDFS(t))

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
for (F,TP) in [(:set_type, :ConfigEnumerator), (:sampler_type, :ConfigSampler)]
    @eval begin
        function $F(::Type{T}, n::Int, nflavor::Int) where {OT, K, T<:TruncatedPoly{K,C,OT} where C}
            TruncatedPoly{K, $F(n,nflavor),OT}
        end
        function $F(::Type{T}, n::Int, nflavor::Int) where {TX, T<:Polynomial{C,TX} where C}
            Polynomial{$F(n,nflavor),:x}
        end
        function $F(::Type{T}, n::Int, nflavor::Int) where {TV, T<:CountingTropical{TV}}
            CountingTropical{TV, $F(n,nflavor)}
        end
        function $F(::Type{Real}, n::Int, nflavor::Int) where {TV}
            $F(n, nflavor)
        end
        function $F(n::Integer, nflavor::Integer)
            s = ceil(Int, log2(nflavor))
            c = _nints(n,s)
            return $TP{n,s,c}
        end
    end
end

# utilities for creating onehot vectors
onehotv(::Type{ConfigEnumerator{N,S,C}}, i::Integer, v) where {N,S,C} = ConfigEnumerator([onehotv(StaticElementVector{N,S,C}, i, v)])
onehotv(::Type{TreeConfigEnumerator{N,S,C}}, i::Integer, v) where {N,S,C} = TreeConfigEnumerator([TreeConfigEnumerator(onehotv(StaticElementVector{N,S,C}, i, v))])
onehotv(::Type{ConfigSampler{N,S,C}}, i::Integer, v) where {N,S,C} = ConfigSampler(onehotv(StaticElementVector{N,S,C}, i, v))
Base.transpose(c::ConfigEnumerator) = c
Base.copy(c::ConfigEnumerator) = ConfigEnumerator(copy(c.data))
Base.transpose(c::TreeConfigEnumerator) = c
Base.copy(c::TreeConfigEnumerator) = isleaf(c) ? TreeConfigEnumerator(c.data) : TreeConfigEnumerator(c.siblings)

# Handle boolean, this is a patch for CUDA matmul
for TYPE in [:ConfigEnumerator, :ConfigSampler, :TruncatedPoly, :TreeConfigEnumerator]
    @eval Base.:*(a::Bool, y::$TYPE) = a ? y : zero(y)
    @eval Base.:*(y::$TYPE, a::Bool) = a ? y : zero(y)
end
