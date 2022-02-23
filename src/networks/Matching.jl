"""
    Matching{CT<:AbstractEinsum} <: GraphProblem
    Matching(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

The [Vertex matching](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/Matching.html) problem.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct Matching{CT<:AbstractEinsum} <: GraphProblem
    code::CT
    ne::Int
end

function Matching(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    edges = [minmax(e.src,e.dst) for e in Graphs.edges(g)]
    rawcode = EinCode(vcat([[s] for s in edges], # labels for edge tensors
                [[minmax(i,j) for j in neighbors(g, i)] for i in Graphs.vertices(g)]),
                collect(Tuple{Int,Int}, openvertices))
    Matching(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), length(edges))
end

flavors(::Type{<:Matching}) = [0, 1]
get_weights(::Matching, i::Int) = [0, 1]
terms(gp::Matching) = getixsv(gp.code)[1:gp.ne]

function generate_tensors(x::T, m::Matching) where T
    m.ne == 0 && return []
    ixs = getixsv(m.code)
    tensors = []
    for i=1:length(ixs)
        ix = ixs[i]
        if i<=m.ne
            @assert length(ix) == 1
            t = Ref(x) .^ get_weights(m, i) # fx is defined on edges.
        else
            t = match_tensor(T, length(ix))
        end
        push!(tensors, t)
    end
    return add_labels!(tensors, ixs, labels(m))
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

