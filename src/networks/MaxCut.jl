"""
    MaxCut{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    MaxCut(graph; weights=UnWeighted(), openvertices=(),
                optimizer=GreedyMethod(), simplifier=nothing)

The [cutting](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/MaxCut.html) problem (or spin glass problem).
In the constructor, `weights` are the weights of edges.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct MaxCut{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    nv::Int
    weights::WT
end
function MaxCut(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == ne(g)
    rawcode = EinCode([[minmax(e.src,e.dst)...] for e in Graphs.edges(g)], collect(Int, openvertices))  # labels for edge tensors
    MaxCut(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), nv(g), weights)
end

flavors(::Type{<:MaxCut}) = [0, 1]
get_weights(gp::MaxCut, i::Int) = [0, gp.weights[i]]
terms(gp::MaxCut) = getixsv(gp.code)
labels(gp::MaxCut) = [1:gp.nv...]

function generate_tensors(x::T, gp::MaxCut) where T
    ixs = getixsv(gp.code)
    return add_labels!(map(enumerate(ixs)) do (i, ix)
        maxcutb((Ref(x) .^ get_weights(gp, i)) ...)
    end, ixs, labels(gp))
end
function maxcutb(a, b)
    return [a b; b a]
end

"""
    cut_size(g::SimpleGraph, config; weights=UnWeighted())

Compute the cut size from vertex `config` (an iterator).
"""
function cut_size(g::SimpleGraph, config; weights=UnWeighted())
    size = zero(eltype(weights)) * false
    for (i, e) in enumerate(edges(g))
        size += (config[e.src] != config[e.dst]) * weights[i]
    end
    return size
end