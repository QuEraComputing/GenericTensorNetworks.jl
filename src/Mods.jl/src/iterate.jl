function Base.iterate(::Type{Mod{m}}) where {m}
    return Mod{m}(0), 0
end

function Base.iterate(::Type{Mod{m}}, s) where {m}
    if s == m - 1
        return nothing
    end
    s += 1
    return Mod{m}(s), s
end

Base.IteratorSize(::Type{Mod{m}}) where {m} = Base.HasLength()

Base.length(::Type{Mod{m}}) where {m} = m




function Base.iterate(::Type{GaussMod{m}}) where {m}
    return GaussMod{m}(0), 0
end

function Base.iterate(::Type{GaussMod{m}}, s) where {m}
    if s == m * m - 1
        return nothing
    end
    s += 1
    a, b = divrem(s, m)
    return GaussMod{m}(a + b * im), s
end


Base.IteratorSize(::Type{GaussMod{m}}) where {m} = Base.HasLength()

Base.length(::Type{GaussMod{m}}) where {m} = m * m
