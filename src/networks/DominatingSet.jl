"""
$TYPEDEF
    DominatingSet(graph; weights=NoWeight())

The [dominating set](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/DominatingSet/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the vertices of the `graph`, default to `NoWeight()`.
"""
struct DominatingSet{WT<:Union{NoWeight, Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    weights::WT
    function DominatingSet(g::SimpleGraph, weights::Union{NoWeight, Vector}=NoWeight())
        @assert weights isa NoWeight || length(weights) == nv(g)
        new{typeof(weights)}(g, weights)
    end
end

function GenericTensorNetwork(problem::DominatingSet; openvertices=(), fixedvertices=Dict{Int,Int}())
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    return GenericTensorNetwork(problem, rawcode, Dict{Int,Int}(fixedvertices))
end
flavors(::Type{<:DominatingSet}) = [0, 1]
terms(gp::DominatingSet) = getixsv(gp.code)
labels(gp::DominatingSet) = [1:length(getixsv(gp.code))...]

# weights interface
get_weights(c::DominatingSet) = c.weights
get_weights(gp::DominatingSet, i::Int) = [0, gp.weights[i]]
chweights(c::DominatingSet, weights) = DominatingSet(c.code, c.graph, weights, c.fixedvertices)

function generate_tensors(x::T, mi::GenericTensorNetwork{<:DominatingSet}) where T
    ixs = getixsv(mi.code)
    isempty(ixs) && return []
	return select_dims(add_labels!(map(enumerate(ixs)) do (i, ix)
        dominating_set_tensor(_pow.(Ref(x), get_weights(mi, i))..., length(ix))
    end, ixs, labels(mi)), ixs, fixedvertices(mi))
end
function dominating_set_tensor(a::T, b::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = a
    end
    t[1<<(d-1)+1:end] .= Ref(b)
    return t
end

"""
    is_dominating_set(g::SimpleGraph, config)

Return true if `config` (a vector of boolean numbers as the mask of vertices) is a dominating set of graph `g`.
"""
is_dominating_set(g::SimpleGraph, config) = all(w->config[w] == 1 || any(v->!iszero(config[v]), neighbors(g, w)), Graphs.vertices(g))