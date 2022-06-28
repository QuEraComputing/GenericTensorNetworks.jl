"""
    MaximalIS{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    MaximalIS(graph; weights=NoWeight(), openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )

The [maximal independent set](https://psychic-meme-f4d866f8.pages.github.io/dev/generated/MaximalIS.html) problem. In the constructor, `weights` are the weights of vertices.

Positional arguments
-------------------------------
* `graph` is the problem graph.

Keyword arguments
-------------------------------
* `weights` are associated with the vertices of the `graph`.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of degree of freedoms, where a value can be `0` (absent in the set) or `1` (present in the set).
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.
"""
struct MaximalIS{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    graph::SimpleGraph
    weights::WT
    fixedvertices::Dict{Int,Int}
end

function MaximalIS(g::SimpleGraph; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing, fixedvertices=Dict{Int,Int}())
    @assert weights isa NoWeight || length(weights) == nv(g)
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    MaximalIS(_optimize_code(rawcode, uniformsize_fix(rawcode, 2, fixedvertices), optimizer, simplifier), g, weights, Dict{Int,Int}(fixedvertices))
end

flavors(::Type{<:MaximalIS}) = [0, 1]
terms(gp::MaximalIS) = getixsv(gp.code)
labels(gp::MaximalIS) = [1:length(getixsv(gp.code))...]
fixedvertices(gp::MaximalIS) = gp.fixedvertices

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