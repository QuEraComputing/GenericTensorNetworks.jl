"""
    MaxCut{CT<:AbstractEinsum,WT<:Union{NoWieght, Vector}} <: GraphProblem
    MaxCut(graph; weights=NoWeight(), openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )

The [cutting](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/MaxCut.html) problem (or spin glass problem).

Positional arguments
-------------------------------
* `graph` is the problem graph.

Keyword arguments
-------------------------------
* `weights` are associated with the edges of the `graph`.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of degree of freedoms, where a value can be `0` (in one side of the cut) or `1` (in the other side of the cut).
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.
"""
struct MaxCut{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    graph::SimpleGraph{Int}
    weights::WT
    fixedvertices::Dict{Int,Int}
end
function MaxCut(g::SimpleGraph; weights=NoWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa NoWeight || length(weights) == ne(g)
    rawcode = EinCode([[minmax(e.src,e.dst)...] for e in Graphs.edges(g)], collect(Int, openvertices))  # labels for edge tensors
    MaxCut(_optimize_code(rawcode, uniformsize_fix(rawcode, 2, fixedvertices), optimizer, simplifier), g, weights, fixedvertices)
end

flavors(::Type{<:MaxCut}) = [0, 1]
get_weights(gp::MaxCut, i::Int) = [0, gp.weights[i]]
terms(gp::MaxCut) = getixsv(gp.code)
labels(gp::MaxCut) = [1:nv(gp.graph)...]
fixedvertices(gp::MaxCut) = gp.fixedvertices

function generate_tensors(x::T, gp::MaxCut) where T
    ixs = getixsv(gp.code)
    tensors = map(enumerate(ixs)) do (i, ix)
        maxcutb((Ref(x) .^ get_weights(gp, i)) ...)
    end
    return select_dims(add_labels!(tensors, ixs, labels(gp)), ixs, fixedvertices(gp))
end

function maxcutb(a, b)
    return [a b; b a]
end

"""
    cut_size(g::SimpleGraph, config; weights=NoWeight())

Compute the cut size from vertex `config` (an iterator).
"""
function cut_size(g::SimpleGraph, config; weights=NoWeight())
    size = zero(eltype(weights)) * false
    for (i, e) in enumerate(edges(g))
        size += (config[e.src] != config[e.dst]) * weights[i]
    end
    return size
end