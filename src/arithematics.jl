export is_commutative_semiring

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

export Max2Poly

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

