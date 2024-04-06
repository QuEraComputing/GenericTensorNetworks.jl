"""
$TYPEDEF

The [maximal independent set](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/MaximalIS/) problem. In the constructor, `weights` are the weights of vertices.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the vertices of the `graph`.
"""
struct MaximalIS{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    graph::SimpleGraph
    weights::WT
    function MaximalIS(g::SimpleGraph, weights::Union{NoWeight, Vector}=NoWeight())
        @assert weights isa NoWeight || length(weights) == nv(g)
        new{typeof(weights)}(g, weights)
    end
end

function GenericTensorNetwork(problem::MaximalIS; openvertices=(), fixedvertices=Dict{Int,Int}())
    rawcode = EinCode(([[Graphs.neighbors(problem.graph, v)..., v] for v in Graphs.vertices(problem.graph)]...,), collect(Int, openvertices))
    return GenericTensorNetwork(problem, rawcode, Dict{Int,Int}(fixedvertices))
end

flavors(::Type{<:MaximalIS}) = [0, 1]
terms(gp::MaximalIS) = getixsv(gp.code)
labels(gp::MaximalIS) = [1:length(getixsv(gp.code))...]

# weights interface
get_weights(c::MaximalIS) = c.weights
get_weights(gp::MaximalIS, i::Int) = [0, gp.weights[i]]
chweights(c::MaximalIS, weights) = MaximalIS(c.code, c.graph, weights, c.fixedvertices)

function generate_tensors(x::T, mi::MaximalIS) where T
    ixs = getixsv(mi.code)
    isempty(ixs) && return []
	return select_dims(add_labels!(map(enumerate(ixs)) do (i, ix)
        maximal_independent_set_tensor(_pow.(Ref(x), get_weights(mi, i))..., length(ix))
    end, ixs, labels(mi)), ixs, fixedvertices(mi))
end
function maximal_independent_set_tensor(a::T, b::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = a
    end
    t[1<<(d-1)+1] = b
    return t
end


"""
    is_maximal_independent_set(g::SimpleGraph, config)

Return true if `config` (a vector of boolean numbers as the mask of vertices) is a maximal independent set of graph `g`.
"""
is_maximal_independent_set(g::SimpleGraph, config) = !any(e->config[e.src] == 1 && config[e.dst] == 1, edges(g)) && all(w->config[w] == 1 || any(v->!iszero(config[v]), neighbors(g, w)), Graphs.vertices(g))