"""
$(TYPEDEF)
    Coloring{K}(graph; weights=UnitWeight())

The [Vertex Coloring](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Coloring/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the edges of the `graph`, default to `UnitWeight()`.
"""
struct Coloring{K, WT<:Union{UnitWeight, Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    weights::WT
    function Coloring{K}(graph::SimpleGraph, weights::Union{UnitWeight, Vector}=UnitWeight()) where {K}
        @assert weights isa UnitWeight || length(weights) == ne(graph)
        new{K, typeof(weights)}(graph, weights)
    end
end
function coloring_network(K::Int, graph::SimpleGraph; weights=UnitWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=MergeVectors())
    cfg = Coloring{K}(graph, weights)
    gtn = GenericTensorNetwork(cfg; openvertices, fixedvertices)
    return OMEinsum.optimize_code(gtn; optimizer, simplifier)
end

flavors(::Type{<:Coloring{K}}) where K = collect(0:K-1)
energy_terms(gp::Coloring) = [[i] for i in 1:nv(gp.graph)]
energy_tensors(::Type{T}, c::Coloring{K}) where {K,T} = [coloringv(T, K) for i=1:nv(c.graph)]
extra_terms(gp::Coloring) = [[minmax(e.src,e.dst)...] for e in Graphs.edges(gp.graph)]
labels(gp::Coloring) = [1:nv(gp.graph)...]

# weights interface
get_weights(c::Coloring) = c.weights
get_weights(c::Coloring{K}, i::Int) where K = fill(c.weights[i], K)
chweights(c::Coloring{K}, weights) where K = Coloring{K}(c.graph, weights)


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
