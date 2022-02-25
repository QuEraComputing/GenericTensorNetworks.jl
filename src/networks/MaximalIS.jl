"""
    MaximalIS{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    MaximalIS(graph; weights=NoWeight(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

The [maximal independent set](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/MaximalIS.html) problem. In the constructor, `weights` are the weights of vertices.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct MaximalIS{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    weights::WT
end

function MaximalIS(g::SimpleGraph; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa NoWeight || length(weights) == nv(g)
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    MaximalIS(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), weights)
end

flavors(::Type{<:MaximalIS}) = [0, 1]
get_weights(gp::MaximalIS, i::Int) = [0, gp.weights[i]]
terms(gp::MaximalIS) = getixsv(gp.code)
labels(gp::MaximalIS) = [1:length(getixsv(gp.code))...]

function generate_tensors(x::T, mi::MaximalIS) where T
    ixs = getixsv(mi.code)
    isempty(ixs) && return []
	return add_labels!(map(enumerate(ixs)) do (i, ix)
        neighbortensor((Ref(x) .^ get_weights(mi, i))..., length(ix))
    end, ixs, labels(mi))
end
function neighbortensor(a::T, b::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = one(T)
    end
    t[1<<(d-1)+1] = a
    t[1<<(d-1)+1] = b
    return t
end


"""
    is_maximal_independent_set(g::SimpleGraph, config)

Return true if `config` (a vector of boolean numbers as the mask of vertices) is a maximal independent set of graph `g`.
"""
is_maximal_independent_set(g::SimpleGraph, config) = !any(e->config[e.src] == 1 && config[e.dst] == 1, edges(g)) && all(w->config[w] == 1 || any(v->!iszero(config[v]), neighbors(g, w)), Graphs.vertices(g))