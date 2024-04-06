"""
$TYPEDEF

The [cutting](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/MaxCut/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `edge_weights` are associated with the edges of the `graph`.
* `vertex_weights` are associated with the vertices of the `graph`.
"""
struct MaxCut{WT1<:Union{NoWeight, Vector},WT2<:Union{ZeroWeight, Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    edge_weights::WT1
    vertex_weights::WT2
    function MaxCut(g::SimpleGraph;
            edge_weights::Union{NoWeight, Vector}=NoWeight(),
            vertex_weights::Union{ZeroWeight, Vector}=ZeroWeight())
        @assert edge_weights isa NoWeight || length(edge_weights) == ne(g)
        @assert vertex_weights isa ZeroWeight || length(vertex_weights) == nv(g)
        new{typeof(edge_weights), typeof(vertex_weights)}(g, edge_weights, vertex_weights)
    end
end
function GenericTensorNetwork(problem::MaxCut; openvertices=(), fixedvertices=Dict{Int,Int}())
    rawcode = EinCode([
        map(e->[minmax(e.src,e.dst)...], Graphs.edges(problem.graph))...,
        map(v->[v], Graphs.vertices(problem.graph))...,
    ], collect(Int, openvertices))  # labels for edge tensors
    return GenericTensorNetwork(problem, rawcode, Dict{Int,Int}(fixedvertices))
end

flavors(::Type{<:MaxCut}) = [0, 1]
# first `ne` indices are for edge weights, last `nv` indices are for vertex weights.
terms(gp::MaxCut) = getixsv(gp.code)
labels(gp::MaxCut) = [1:nv(gp.graph)...]
fixedvertices(gp::MaxCut) = gp.fixedvertices

# weights interface
get_weights(c::MaxCut) = [[c.edge_weights[i] for i=1:ne(c.graph)]..., [c.vertex_weights[i] for i=1:nv(c.graph)]...]
get_weights(gp::MaxCut, i::Int) = i <= ne(gp.graph) ? [0, gp.edge_weights[i]] : [0, gp.vertex_weights[i-ne(gp.graph)]]
chweights(c::MaxCut, weights) = MaxCut(c.code, c.graph, weights[1:ne(c.graph)], weights[ne(c.graph)+1:end], c.fixedvertices)

function generate_tensors(x::T, gp::MaxCut) where T
    ixs = getixsv(gp.code)
    l = ne(gp.graph)
    tensors = [
        Array{T}[maxcutb(_pow.(Ref(x), get_weights(gp, i))...) for i=1:l]...,
        add_labels!(Array{T}[Ref(x) .^ get_weights(gp, i+l) for i=1:nv(gp.graph)], ixs[l+1:end], labels(gp))...
    ]
    return select_dims(tensors, ixs, fixedvertices(gp))
end

function maxcutb(a, b)
    return [a b; b a]
end

"""
    cut_size(g::SimpleGraph, config; edge_weights=NoWeight(), vertex_weights=ZeroWeight())

Compute the cut size for the vertex configuration `config` (an iterator).
"""
function cut_size(g::SimpleGraph, config; edge_weights=NoWeight(), vertex_weights=ZeroWeight())
    size = zero(promote_type(eltype(vertex_weights), eltype(edge_weights)))
    for (i, e) in enumerate(edges(g))
        size += (config[e.src] != config[e.dst]) * edge_weights[i]
    end
    for v in vertices(g)
        size += config[v] * vertex_weights[v]
    end
    return size
end
