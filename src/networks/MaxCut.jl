"""
$TYPEDEF

The [cutting](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/MaxCut/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `edge_weights` are associated with the edges of the `graph`.
* `vertex_weights` are associated with the vertices of the `graph`.
"""
struct MaxCut{WT1<:Union{UnitWeight, Vector},WT2<:Union{ZeroWeight, Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    edge_weights::WT1
    vertex_weights::WT2
    function MaxCut(g::SimpleGraph,
            edge_weights::Union{UnitWeight, Vector}=UnitWeight(),
            vertex_weights::Union{ZeroWeight, Vector}=ZeroWeight())
        @assert edge_weights isa UnitWeight || length(edge_weights) == ne(g)
        @assert vertex_weights isa ZeroWeight || length(vertex_weights) == nv(g)
        new{typeof(edge_weights), typeof(vertex_weights)}(g, edge_weights, vertex_weights)
    end
end

flavors(::Type{<:MaxCut}) = [0, 1]
# first `ne` indices are for edge weights, last `nv` indices are for vertex weights.
energy_terms(gp::MaxCut) = [[[minmax(e.src,e.dst)...] for e in Graphs.edges(gp.graph)]...,
                            [[v] for v in Graphs.vertices(gp.graph)]...]
energy_tensors(x::T, c::MaxCut) where T = [[maxcutb(_pow.(Ref(x), get_weights(c, i))...) for i=1:ne(c.graph)]...,
                                            [Ref(x) .^ get_weights(c, i+ne(c.graph)) for i=1:nv(c.graph)]...]
extra_terms(::MaxCut) = Vector{Int}[]
extra_tensors(::Type{T}, ::MaxCut) where T = Array{T}[]
labels(gp::MaxCut) = [1:nv(gp.graph)...]

# weights interface
get_weights(c::MaxCut) = [[c.edge_weights[i] for i=1:ne(c.graph)]..., [c.vertex_weights[i] for i=1:nv(c.graph)]...]
get_weights(gp::MaxCut, i::Int) = i <= ne(gp.graph) ? [0, gp.edge_weights[i]] : [0, gp.vertex_weights[i-ne(gp.graph)]]
chweights(c::MaxCut, weights) = MaxCut(c.graph, weights[1:ne(c.graph)], weights[ne(c.graph)+1:end])

function maxcutb(a, b)
    return [a b; b a]
end

"""
    cut_size(g::SimpleGraph, config; edge_weights=UnitWeight(), vertex_weights=ZeroWeight())

Compute the cut size for the vertex configuration `config` (an iterator).
"""
function cut_size(g::SimpleGraph, config; edge_weights=UnitWeight(), vertex_weights=ZeroWeight())
    size = zero(promote_type(eltype(vertex_weights), eltype(edge_weights)))
    for (i, e) in enumerate(edges(g))
        size += (config[e.src] != config[e.dst]) * edge_weights[i]
    end
    for v in vertices(g)
        size += config[v] * vertex_weights[v]
    end
    return size
end
