"""
    Matching{CT<:AbstractEinsum} <: GraphProblem
    Matching(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

[Vertex matching](https://mathworld.wolfram.com/Matching.html) problem.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct Matching{CT<:AbstractEinsum} <: GraphProblem
    symbols::Vector{Tuple{Int,Int}}
    code::CT
end

function Matching(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    symbols = [minmax(e.src,e.dst) for e in Graphs.edges(g)]
    rawcode = EinCode(vcat([[s] for s in symbols], # labels for edge tensors
                [[minmax(i,j) for j in neighbors(g, i)] for i in Graphs.vertices(g)]),
                collect(Tuple{Int,Int}, openvertices))
    Matching(symbols, _optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier))
end

flavors(::Type{<:Matching}) = [0, 1]
get_weights(::Matching, sym) = [0, 1]
symbols(m::Matching) = m.symbols

function generate_tensors(fx, m::Matching)
    syms = symbols(m)
    isempty(syms) && return []
    ixs = getixsv(m.code)
    T = eltype(fx(syms[1]))
    n = length(syms)  # number of vertices
    tensors = []
    for i=1:length(ixs)
        if i<=n
            @assert length(ixs[i]) == 1
            t = fx(syms[i]) # fx is defined on edges.
        else
            t = match_tensor(T, length(ixs[i]))
        end
        push!(tensors, t)
    end
    return tensors
end
function match_tensor(::Type{T}, n::Int) where T
    t = zeros(T, fill(2, n)...)
    for ci in CartesianIndices(t)
        if sum(ci.I .- 1) <= 1
            t[ci] = one(T)
        end
    end
    return t
end

