using Polynomials: Polynomial
using TropicalNumbers: Tropical, CountingTropical
using Mods, Primes
using Base.Cartesian
import AbstractTrees: children, printnode, print_tree

@enum TreeTag LEAF SUM PROD ZERO ONE

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

######################## Truncated Polynomial ######################
# TODO: store orders to support non-integer weights
# (↑) Maybe not so nessesary, no use case for counting degeneracy when using floating point weights.
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
Base.:(==)(t1::TruncatedPoly{K}, t2::TruncatedPoly{K}) where K = t1.maxorder == t2.maxorder && all(i->t1.coeffs[i] == t2.coeffs[i], 1:K)

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

############################ ExtendedTropical #####################
"""
    ExtendedTropical{K,TO} <: Number
    ExtendedTropical{K}(orders)

Extended Tropical numbers with largest `K` orders keeped,
or the [`TruncatedPoly`](@ref) without coefficients,
`TO` is the element type of orders, usually [`Tropical`](@ref) numbers.
This algebra maps

* `+` to finding largest `K` values of union of two sets.
* `*` to finding largest `K` values of sum combination of two sets.
* `0` to set [-Inf, -Inf, ..., -Inf, -Inf]
* `1` to set [-Inf, -Inf, ..., -Inf, 0]

Example
------------------------------
```jldoctest; setup=(using GraphTensorNetworks)
julia> x = ExtendedTropical{3}(Tropical.([1.0, 2, 3]))
ExtendedTropical{3, TropicalF64}(TropicalF64[1.0ₜ, 2.0ₜ, 3.0ₜ])

julia> y = ExtendedTropical{3}(Tropical.([-Inf, 2, 5]))
ExtendedTropical{3, TropicalF64}(TropicalF64[-Infₜ, 2.0ₜ, 5.0ₜ])

julia> x * y
ExtendedTropical{3, TropicalF64}(TropicalF64[6.0ₜ, 7.0ₜ, 8.0ₜ])

julia> x + y
ExtendedTropical{3, TropicalF64}(TropicalF64[2.0ₜ, 3.0ₜ, 5.0ₜ])

julia> one(x)
ExtendedTropical{3, TropicalF64}(TropicalF64[-Infₜ, -Infₜ, 0.0ₜ])

julia> zero(x)
ExtendedTropical{3, TropicalF64}(TropicalF64[-Infₜ, -Infₜ, -Infₜ])
```
"""
struct ExtendedTropical{K,TO} <: Number
    orders::Vector{TO}
end
function ExtendedTropical{K}(x::Vector{T}) where {T, K}
    @assert length(x) == K
    @assert issorted(x)
    ExtendedTropical{K,T}(x)
end
Base.:(==)(a::ExtendedTropical{K}, b::ExtendedTropical{K}) where K = all(i->a.orders[i] == b.orders[i], 1:K)

function Base.:*(a::ExtendedTropical{K,TO}, b::ExtendedTropical{K,TO}) where {K,TO}
    res = Vector{TO}(undef, K)
    return ExtendedTropical{K,TO}(sorted_sum_combination!(res, a.orders, b.orders))
end

# 1. bisect over summed value and find the critical value `c`,
# 2. collect the values with sum combination `≥ c`,
# 3. sort the collected values
function sorted_sum_combination!(res::AbstractVector{TO}, A::AbstractVector{TO}, B::AbstractVector{TO}) where TO
    K = length(res)
    @assert length(B) == length(A) == K
    @inbounds high = A[K] * B[K]

    mA = findfirst(!iszero, A)
    mB = findfirst(!iszero, B)
    if mA === nothing || mB === nothing
        res .= Ref(zero(TO))
        return res
    end
    @inbounds low = A[mA] * B[mB]
    # count number bigger than x
    c, _ = count_geq(A, B, mB, low, true)
    @inbounds if c <= K   # return
        res[K-c+1:K] .= sort!(collect_geq!(view(res,1:c), A, B, mB, low))
        if c < K
            res[1:K-c] .= zero(TO)
        end
        return res
    end
    # calculate by bisection for at most 30 times.
    @inbounds for _ = 1:30
        mid = mid_point(high, low)
        c, nB = count_geq(A, B, mB, mid, true)
        if c > K
            low = mid
            mB = nB
        elseif c == K  # return
            # NOTE: this is the bottleneck
            return sort!(collect_geq!(res, A, B, mB, mid))
        else
            high = mid
        end
    end
    clow, _ = count_geq(A, B, mB, low, false)
    @inbounds res .= sort!(collect_geq!(similar(res, clow), A, B, mB, low))[end-K+1:end]
    return res
end

# count the number of sum-combinations with the sum >= low
function count_geq(A, B, mB, low, earlybreak)
    K = length(A)
    k = 1   # TODO: we should tighten mA, mB later!
    @inbounds Ak = A[K-k+1]
    @inbounds Bq = B[K-mB+1]
    c = 0
    nB = mB
    @inbounds for q = K-mB+1:-1:1
        Bq = B[K-q+1]
        while k < K && Ak * Bq >= low
            k += 1
            Ak = A[K-k+1]
        end
        if Ak * Bq >= low
            c += k
        else
            c += (k-1)
            if k==1
                nB += 1
            end
        end
        if earlybreak && c > K
            return c, nB
        end
    end
    return c, nB
end

function collect_geq!(res, A, B, mB, low)
    K = length(A)
    k = 1   # TODO: we should tighten mA, mB later!
    Ak = A[K-k+1]
    l = 0
    for q = K-mB+1:-1:1
        Bq = B[K-q+1]
        while k < K && Ak * Bq >= low
            k += 1
            Ak = A[K-k+1]
        end
        # push data
        ck = Ak * Bq >= low ? k : k-1
        for j=1:ck
            l += 1
            res[l] = Bq * A[end-j+1]
        end
    end
    return res
end

# for bisection
mid_point(a::Tropical{T}, b::Tropical{T}) where T = Tropical{T}((a.n + b.n) / 2)
mid_point(a::CountingTropical{T,CT}, b::CountingTropical{T,CT}) where {T,CT} = CountingTropical{T,CT}((a.n + b.n) / 2, a.c)
mid_point(a::Tropical{T}, b::Tropical{T}) where T<:Integer = Tropical{T}((a.n + b.n) ÷ 2)
mid_point(a::CountingTropical{T,CT}, b::CountingTropical{T,CT}) where {T<:Integer,CT} = CountingTropical{T,CT}((a.n + b.n) ÷ 2, a.c)

function Base.:+(a::ExtendedTropical{K,TO}, b::ExtendedTropical{K,TO}) where {K,TO}
    res = Vector{TO}(undef, K)
    ptr1, ptr2 = K, K
    @inbounds va, vb = a.orders[ptr1], b.orders[ptr2]
    @inbounds for i=K:-1:1
        if va > vb
            res[i] = va
            if ptr1 != 1
                ptr1 -= 1
                va = a.orders[ptr1]
            end
        else
            res[i] = vb
            if ptr2 != 1
                ptr2 -= 1
                vb = b.orders[ptr2]
            end
        end
    end
    return ExtendedTropical{K,TO}(res)
end

Base.:^(a::ExtendedTropical, b::Integer) = Base.invoke(^, Tuple{ExtendedTropical, Real}, a, b)
function Base.:^(a::ExtendedTropical{K,TO}, b::Real) where {K,TO}
    if iszero(b)  # to avoid NaN
        return one(ExtendedTropical{K,TO})
    else
        return ExtendedTropical{K,TO}(a.orders .^ b)
    end
end

Base.zero(::Type{ExtendedTropical{K,TO}}) where {K,TO} = ExtendedTropical{K,TO}(fill(zero(TO), K))
Base.one(::Type{ExtendedTropical{K,TO}}) where {K,TO} = ExtendedTropical{K,TO}(map(i->i==K ? one(TO) : zero(TO), 1:K))
Base.zero(::ExtendedTropical{K,TO}) where {K,TO} = zero(ExtendedTropical{K,TO})
Base.one(::ExtendedTropical{K,TO}) where {K,TO} = one(ExtendedTropical{K,TO})

############################ SET Numbers ##########################
abstract type AbstractSetNumber end

"""
    ConfigEnumerator{N,S,C} <: AbstractSetNumber

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
struct ConfigEnumerator{N,S,C} <: AbstractSetNumber
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
    ConfigSampler{N,S,C} <: AbstractSetNumber
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
struct ConfigSampler{N,S,C} <: AbstractSetNumber
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

"""
    SumProductTree{ET} <: AbstractSetNumber

Configuration enumerator encoded in a tree, it is the most natural representation given by a sum-product network
and is often more memory efficient than putting the configurations in a vector.
One can use [`generate_samples`](@ref) to sample configurations from this tree structure efficiently.

Fields
-----------------------
* `tag` is one of `ZERO`, `ONE`, `LEAF`, `SUM`, `PROD`.
* `data` is the element stored in a `LEAF` node.
* `left` and `right` are two operands of a `SUM` or `PROD` node.

Example
------------------------
```jldoctest; setup=:(using GraphTensorNetworks)
julia> s = SumProductTree(bv"00111")
00111


julia> q = SumProductTree(bv"10000")
10000


julia> x = s + q
+
├─ 00111
└─ 10000


julia> y = x * x
*
├─ +
│  ├─ 00111
│  └─ 10000
└─ +
   ├─ 00111
   └─ 10000


julia> collect(y)
4-element Vector{StaticBitVector{5, 1}}:
 00111
 10111
 10111
 10000

julia> zero(s)
∅



julia> one(s)
00000


```
"""
mutable struct SumProductTree{ET} <: AbstractSetNumber
    tag::TreeTag
    data::ET
    left::SumProductTree{ET}
    right::SumProductTree{ET}
    # zero(ET) can be undef
    function SumProductTree(tag::TreeTag, left::SumProductTree{ET}, right::SumProductTree{ET}) where {ET}
        res = new{ET}(tag)
        res.left = left
        res.right = right
        return res
    end
    function SumProductTree(data::ET) where ET
        return new{ET}(LEAF, data)
    end
    function SumProductTree{ET}(tag::TreeTag) where {ET}
        @assert  tag === ZERO || tag === ONE
        return new{ET}(tag)
    end
end
# these two interfaces must be implemented in order to collect elements
_data_mul(x::StaticElementVector, y::StaticElementVector) = x | y
_data_one(::Type{T}) where T<:StaticElementVector = zero(T)  # NOTE: might be optional

"""
    TreeConfigEnumerator{N,S,C}
    
An alias for [`SumProductTree`](@ref)`{StaticElementVector{N, S, C}}`,
which is a useful element type for configuration enumeration.
"""
const TreeConfigEnumerator{N,S,C} = SumProductTree{StaticElementVector{N,S,C}}
TreeConfigEnumerator(data::StaticElementVector) = SumProductTree(data)
TreeConfigEnumerator(tag::TreeTag, left::TreeConfigEnumerator{N,S,C}, right::TreeConfigEnumerator{N,S,C}) where {N,S,C} = SumProductTree(tag, left, right)

# AbstractTree APIs
function children(t::SumProductTree)
    if t.tag == ZERO || t.tag == LEAF || t.tag == ONE
        return typeof(t)[]
    else
        return [t.left, t.right]
    end
end
function printnode(io::IO, t::SumProductTree{ET}) where {ET}
    if t.tag === LEAF
        print(io, t.data)
    elseif t.tag === ZERO
        print(io, "∅")
    elseif t.tag === ONE
        print(io, _data_one(ET))
    elseif t.tag === SUM
        print(io, "+")
    else  # PROD
        print(io, "*")
    end
end

# it must be mutable, otherwise, objectid might be slow serialization might fail.
# IdDict is much slower than Dict, it is useless.
Base.length(x::SumProductTree) = _length!(x, Dict{UInt, Float64}())

function _length!(x, d)
    id = objectid(x)
    haskey(d, id) && return d[id]
    if x.tag === SUM
        l = _length!(x.left, d) + _length!(x.right, d)
        d[id] = l
        return l
    elseif x.tag === PROD
        l = _length!(x.left, d) * _length!(x.right, d)
        d[id] = l
        return l
    elseif x.tag === ZERO
        return 0.0
    else
        return 1.0
    end
end

# # loop version
# function _length!(x, d)
#     rootid = objectid(x)
#     t_stack = [x]
#     # update dict
#     while !isempty(t_stack)
#         x = t_stack[end]
#         id = objectid(x)
#         if haskey(d, id)
#             pop!(t_stack)
#         else
#             if x.tag === SUM
#                 idl = objectid(x.left)
#                 if haskey(d, idl)
#                     idr = objectid(x.right)
#                     if haskey(d, idr)
#                         @inbounds d[id] = d[idl] + d[idr]
#                         pop!(t_stack)
#                     else
#                         push!(t_stack, x.right)
#                     end
#                 else
#                     push!(t_stack, x.left)
#                 end
#             elseif x.tag === PROD
#                 idl = objectid(x.left)
#                 if haskey(d, idl)
#                     idr = objectid(x.right)
#                     if haskey(d, idr)
#                         @inbounds d[id] = d[idl] * d[idr]
#                         pop!(t_stack)
#                     else
#                         push!(t_stack, x.right)
#                     end
#                 else
#                     push!(t_stack, x.left)
#                 end
#             elseif x.tag === ZERO
#                 d[id] = 0.0
#                 pop!(t_stack)
#             else
#                 d[id] = 1.0
#                 pop!(t_stack)
#             end
#         end
#     end
#     return d[rootid]
# end

function _find_branch(x, d)
    if x.tag === ZERO
        return true, 0.0
    elseif x.tag === ONE || x.tag === LEAF
        return true, 1.0
    else
        idl = objectid(x.left)
        if haskey(d, idl)
            return true, d[idl]
        else
            return false, 0.0
        end
    end
end


num_nodes(x::SumProductTree) = _num_nodes(x, Dict{UInt, Int}())
function _num_nodes(x, d)
    id = objectid(x)
    haskey(d, id) && return 0
    if x.tag == ZERO || x.tag == ONE
        res = 1
    elseif x.tag == LEAF
        res = 1
    else
        res = _num_nodes(x.left, d) + _num_nodes(x.right, d) + 1
    end
    d[id] = res
    return res
end

function Base.:(==)(x::SumProductTree{ET}, y::SumProductTree{ET}) where {ET}
    return Set(collect(x)) == Set(collect(y))
end

Base.show(io::IO, t::SumProductTree) = print_tree(io, t)

function Base.collect(x::SumProductTree{ET}) where {ET}
    if x.tag == ZERO
        return ET[]
    elseif x.tag == ONE
        return [_data_one(ET)]
    elseif x.tag == LEAF
        return [x.data]
    elseif x.tag == SUM
        return vcat(collect(x.left), collect(x.right))
    else   # PROD
        return vec([reduce(_data_mul, si) for si in Iterators.product(collect(x.left), collect(x.right))])
    end
end

function Base.:+(x::SumProductTree{ET}, y::SumProductTree{ET}) where {ET}
    if x.tag == ZERO
        return y
    elseif y.tag == ZERO
        return x
    else
        return SumProductTree(SUM, x, y)
    end
end

function Base.:*(x::SumProductTree{ET}, y::SumProductTree{ET}) where {ET}
    if x.tag == ONE
        return y
    elseif y.tag == ONE
        return x
    elseif x.tag == ZERO
        return x
    elseif y.tag == ZERO
        return y
    elseif x.tag == LEAF && y.tag == LEAF
        return SumProductTree(_data_mul(x.data, y.data))
    else
        return SumProductTree(PROD, x, y)
    end
end

Base.zero(::Type{SumProductTree{ET}}) where {ET} = SumProductTree{ET}(ZERO)
Base.one(::Type{SumProductTree{ET}}) where {ET} = SumProductTree{ET}(ONE)
Base.zero(::SumProductTree{ET}) where {ET} = zero(SumProductTree{ET})
Base.one(::SumProductTree{ET}) where {ET} = one(SumProductTree{ET})
# todo, check siblings too?
function Base.iszero(t::SumProductTree)
    if t.tag == SUM
        iszero(t.left) && iszero(t.right)
    elseif t.tag == ZERO
        true
    elseif t.tag == LEAF || t.tag == ONE
        false
    else
        iszero(t.left) || iszero(t.right)
    end
end

"""
    generate_samples(t::SumProductTree, nsamples::Int)

Direct sampling configurations from a [`SumProductTree`](@ref) instance.

Example
-----------------------------
```jldoctest; setup=:(using GraphTensorNetworks)
julia> using Graphs

julia> g= smallgraph(:petersen)
{10, 15} undirected simple Int64 graph

julia> t = solve(IndependentSet(g), ConfigsAll(; tree_storage=true))[];

julia> samples = generate_samples(t, 1000);

julia> all(s->is_independent_set(g, s), samples)
true
```
"""
function generate_samples(t::SumProductTree{ET}, nsamples::Int) where {ET}
    # get length dict
    res = fill(_data_one(ET), nsamples)
    d = Dict{UInt, Float64}()
    sample_descend!(res, t, d)
    return res
end

function sample_descend!(res::AbstractVector, t::SumProductTree, d::Dict)
    res_stack = Any[res]
    t_stack = [t]
    while !isempty(t_stack) && !isempty(res_stack)
        t = pop!(t_stack)
        res = pop!(res_stack)
        if t.tag == LEAF
            res .|= Ref(t.data)
        elseif t.tag == SUM
            ratio = _length!(t.left, d)/_length!(t, d)
            nleft = 0
            for _ = 1:length(res)
                if rand() < ratio
                    nleft += 1
                end
            end
            shuffle!(res)  # shuffle the `res` to avoid biased sampling, very important.
            push!(res_stack, view(res,1:nleft))
            push!(res_stack, view(res,nleft+1:length(res)))
            push!(t_stack, t.left)
            push!(t_stack, t.right)
        elseif t.tag == PROD
            push!(res_stack, res)
            push!(res_stack, res)
            push!(t_stack, t.left)
            push!(t_stack, t.right)
        elseif t.tag == ZERO
            error("Meet zero when descending.")
        else
            # pass for 1
        end
    end
    return res
end

# A patch to make `Polynomial{ConfigEnumerator}` work
function Base.:*(a::Int, y::AbstractSetNumber)
    a == 0 && return zero(y)
    a == 1 && return y
    error("multiplication between int and `$(typeof(y))` is not defined.")
end

# convert from counting type to bitstring type
for (F,TP) in [(:set_type, :ConfigEnumerator), (:sampler_type, :ConfigSampler), (:treeset_type, :TreeConfigEnumerator)]
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
sampler_type(::Type{ExtendedTropical{K,T}}, n::Int, nflavor::Int) where {K,T} = ExtendedTropical{K, sampler_type(T, n, nflavor)}

# utilities for creating onehot vectors
onehotv(::Type{ConfigEnumerator{N,S,C}}, i::Integer, v) where {N,S,C} = ConfigEnumerator([onehotv(StaticElementVector{N,S,C}, i, v)])
# we treat `v == 0` specially because we want the final result not containing one leaves.
onehotv(::Type{TreeConfigEnumerator{N,S,C}}, i::Integer, v) where {N,S,C} = v == 0 ? one(TreeConfigEnumerator{N,S,C}) : TreeConfigEnumerator(onehotv(StaticElementVector{N,S,C}, i, v))
onehotv(::Type{ConfigSampler{N,S,C}}, i::Integer, v) where {N,S,C} = ConfigSampler(onehotv(StaticElementVector{N,S,C}, i, v))
# just to make matrix transpose work
Base.transpose(c::ConfigEnumerator) = c
Base.copy(c::ConfigEnumerator) = ConfigEnumerator(copy(c.data))
Base.transpose(c::SumProductTree) = c
function Base.copy(c::SumProductTree{ET}) where {ET}
    if c.tag == LEAF
        SumProductTree(c.data)
    elseif c.tag == ZERO || c.tag == ONE
        SumProductTree{ET}(c.tag)
    else
        SumProductTree(c.tag, c.left, c.right)
    end
end

# Handle boolean, this is a patch for CUDA matmul
for TYPE in [:AbstractSetNumber, :TruncatedPoly, :ExtendedTropical]
    @eval Base.:*(a::Bool, y::$TYPE) = a ? y : zero(y)
    @eval Base.:*(y::$TYPE, a::Bool) = a ? y : zero(y)
end

# to handle power of polynomials
function Base.:^(x::SumProductTree, y::Real)
    if y == 0
        return one(x)
    elseif x.tag == LEAF || x.tag == ONE || x.tag == ZERO
        return x
    else
        error("pow of non-leaf nodes is forbidden!")
    end
end
function Base.:^(x::ConfigEnumerator, y::Real)
    if y <= 0
        return one(x)
    elseif length(x) <= 1
        return x
    else
        error("pow of configuration enumerator of `size > 1` is forbidden!")
    end
end
function Base.:^(x::ConfigSampler, y::Real)
    if y <= 0
        return one(x)
    else
        return x
    end
end

# variable `x`
function _x(::Type{Polynomial{BS,X}}; invert) where {BS,X}
    @assert !invert   # not supported, because it is not useful
    Polynomial{BS,X}([zero(BS), one(BS)])
end
function _x(::Type{TruncatedPoly{K,BS,OS}}; invert) where {K,BS,OS}
    ret = TruncatedPoly{K,BS,OS}(ntuple(i->i<K ? zero(BS) : one(BS), K),one(OS))
    invert ? pre_invert_exponent(ret) : ret
end
function _x(::Type{CountingTropical{TV,BS}}; invert) where {TV,BS}
    ret = CountingTropical{TV,BS}(one(TV), one(BS))
    invert ? pre_invert_exponent(ret) : ret
end
function _x(::Type{Tropical{TV}}; invert) where {TV}
    ret = Tropical{TV}(one(TV))
    invert ? pre_invert_exponent(ret) : ret
end
function _x(::Type{ExtendedTropical{K,TO}}; invert) where {K,TO}
    return ExtendedTropical{K,TO}(map(i->i==K ? _x(TO; invert=invert) : zero(TO), 1:K))
end

# for finding all solutions
function _x(::Type{T}; invert) where {T<:AbstractSetNumber}
    ret = one(T)
    invert ? pre_invert_exponent(ret) : ret
end

function _onehotv(::Type{Polynomial{BS,X}}, x, v) where {BS,X}
    Polynomial{BS,X}([onehotv(BS, x, v)])
end
function _onehotv(::Type{TruncatedPoly{K,BS,OS}}, x, v) where {K,BS,OS}
    TruncatedPoly{K,BS,OS}(ntuple(i->i != K ? zero(BS) : onehotv(BS, x, v), K),zero(OS))
end
function _onehotv(::Type{CountingTropical{TV,BS}}, x, v) where {TV,BS}
    CountingTropical{TV,BS}(zero(TV), onehotv(BS, x, v))
end
function _onehotv(::Type{BS}, x, v) where {BS<:AbstractSetNumber}
    onehotv(BS, x, v)
end
function _onehotv(::Type{ExtendedTropical{K,TO}}, x, v) where {K,T,BS<:AbstractSetNumber,TO<:CountingTropical{T,BS}}
    ExtendedTropical{K,TO}(map(i->i==K ? _onehotv(TO, x, v) : zero(TO), 1:K))
end

# negate the exponents before entering the solver
pre_invert_exponent(t::TruncatedPoly{K}) where K = TruncatedPoly(t.coeffs, -t.maxorder)
pre_invert_exponent(t::TropicalNumbers.TropicalTypes) = inv(t)
# negate the exponents after entering the solver
post_invert_exponent(t::TruncatedPoly{K}) where K = TruncatedPoly(ntuple(i->t.coeffs[K-i+1], K), -t.maxorder+(K-1))
post_invert_exponent(t::TropicalNumbers.TropicalTypes) = inv(t)
post_invert_exponent(t::ExtendedTropical{K}) where K = ExtendedTropical{K}(map(i->inv(t.orders[i]), K:-1:1))
