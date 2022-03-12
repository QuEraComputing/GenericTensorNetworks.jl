"""
    Coloring{K,CT<:AbstractEinsum, WT<:Union{NoWeight, Vector}} <: GraphProblem
    Coloring{K}(graph; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

The [Vertex Coloring](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/Coloring.html) problem.
`weights` has one to one correspondence with `edges(graph)`.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct Coloring{K,CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    graph::SimpleGraph{Int}
    weights::WT
end
Coloring{K}(code::ET, graph::SimpleGraph, weights::Union{NoWeight, Vector}) where {K,ET<:AbstractEinsum} = Coloring{K,ET,typeof(weights)}(code, graph, weights)
# same network layout as independent set.
function Coloring{K}(g::SimpleGraph; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where K
    @assert weights isa NoWeight || length(weights) == ne(g)
    rawcode = EinCode(([[i] for i in Graphs.vertices(g)]..., # labels for vertex tensors
                       [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]...), collect(Int, openvertices))  # labels for edge tensors
    code = _optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier)
    Coloring{K}(code, g, weights)
end

flavors(::Type{<:Coloring{K}}) where K = collect(0:K-1)
get_weights(c::Coloring{K}, i::Int) where K = fill(c.weights[i], K)
terms(gp::Coloring) = getixsv(gp.code)[1:nv(gp.graph)]
labels(gp::Coloring) = [1:nv(gp.graph)...]

function generate_tensors(x::T, c::Coloring{K}) where {K,T}
    ixs = getixsv(c.code)
    return vcat(add_labels!([coloringv(T, K) for i=1:nv(c.graph)], ixs[1:nv(c.graph)], labels(c)), [coloringb(x, K) .^ get_weights(c, i) for i=1:ne(c.graph)])
end

# coloring bond tensor
function coloringb(x::T, k::Int) where T
    x = fill(x, k, k)
    for i=1:k
        x[i,i] = one(T)
    end
    return x
end
# coloring vertex tensor
coloringv(::Type{T}, k::Int) where T = fill(one(T), k)

# utilities
"""
    is_good_vertex_coloring(graph::SimpleGraph, config)

Returns true if the coloring specified by config is a valid one, i.e. does not violate the contraints of vertices of an edges having different colors.
"""
function is_good_vertex_coloring(graph::SimpleGraph, config)
    for e in edges(graph)
        config[e.src] == config[e.dst] && return false
    end
    return true
end