"""
$(TYPEDEF)
    Coloring{K}(graph; weights=NoWeight())

The [Vertex Coloring](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Coloring/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the edges of the `graph`, default to `NoWeight()`.
"""
struct Coloring{K, WT<:Union{NoWeight, Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    weights::WT
    function Coloring{K}(graph::SimpleGraph, weights::Union{NoWeight, Vector}=NoWeight()) where {K}
        @assert weights isa NoWeight || length(weights) == ne(g)
        new{K, typeof(weights)}(graph, weights)
    end
end
function GenericTensorNetwork(problem::Coloring{K}; openvertices=(), fixedvertices=Dict{Int,Int}()) where K
    rawcode = EinCode(([[i] for i in Graphs.vertices(g)]..., # labels for vertex tensors
                       [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]...), collect(Int, openvertices))  # labels for edge tensors
    return GenericTensorNetwork(problem, rawcode, Dict{Int,Int}(fixedvertices))
end
flavors(::Type{<:Coloring{K}}) where K = collect(0:K-1)
terms(gp::Coloring) = [[i] for i in 1:nv(gp.graph)]
labels(gp::Coloring) = [1:nv(gp.graph)...]

# weights interface
get_weights(c::Coloring) = c.weights
get_weights(c::Coloring{K}, i::Int) where K = fill(c.weights[i], K)
chweights(c::Coloring{K}, weights) where K = Coloring{K}(c.code, c.graph, weights, c.fixedvertices)

function generate_tensors(x::T, c::GenericTensorNetwork{<:Coloring{K}}) where {K,T}
    ixs = getixsv(c.code)
    graph = c.problem.graph
    return select_dims([
        add_labels!(Array{T}[coloringv(T, K) for i=1:nv(graph)], ixs[1:nv(graph)], labels(c))...,
        Array{T}[_pow.(coloringb(x, K), get_weights(c, i)) for i=1:ne(graph)]...
    ], ixs, fixedvertices(c))
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
    is_vertex_coloring(graph::SimpleGraph, config)

Returns true if the coloring specified by config is a valid one, i.e. does not violate the contraints of vertices of an edges having different colors.
"""
function is_vertex_coloring(graph::SimpleGraph, config)
    for e in edges(graph)
        config[e.src] == config[e.dst] && return false
    end
    return true
end
