"""
    DominatingSet{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    DominatingSet(graph; weights=NoWeight(), openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )

The [dominating set](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/DominatingSet.html) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.

Keyword arguments
-------------------------------
* `weights` are associated with the vertices of the `graph`.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of degree of freedoms on vertices, where a value can be `0` (absent in the set) or `1` (present in the set).
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.
"""
struct DominatingSet{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    graph::SimpleGraph{Int}
    weights::WT
    fixedvertices::Dict{Int,Int}
end

function DominatingSet(g::SimpleGraph; weights=NoWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa NoWeight || length(weights) == nv(g)
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    DominatingSet(_optimize_code(rawcode, uniformsize_fix(rawcode, 2, fixedvertices), optimizer, simplifier), g, weights, fixedvertices)
end

flavors(::Type{<:DominatingSet}) = [0, 1]
get_weights(gp::DominatingSet, i::Int) = [0, gp.weights[i]]
terms(gp::DominatingSet) = getixsv(gp.code)
labels(gp::DominatingSet) = [1:length(getixsv(gp.code))...]
fixedvertices(gp::DominatingSet) = gp.fixedvertices

function generate_tensors(x::T, mi::DominatingSet) where T
    ixs = getixsv(mi.code)
    isempty(ixs) && return []
	return select_dims(add_labels!(map(enumerate(ixs)) do (i, ix)
        dominating_set_tensor((Ref(x) .^ get_weights(mi, i))..., length(ix))
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