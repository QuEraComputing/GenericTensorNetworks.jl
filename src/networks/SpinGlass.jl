"""
    SpinGlass{CT<:AbstractEinsum,WT<:Union{NoWieght, Vector}} <: GraphProblem
    SpinGlass(graph; edge_weights=NoWeight(), vertex_weights=NoWeight(), openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )

The [spin glass](https://psychic-meme-f4d866f8.pages.github.io/dev/generated/SpinGlass.html) problem (or cutting problem).

Positional arguments
-------------------------------
* `graph` is the problem graph.

Keyword arguments
-------------------------------
* `edge_weights` are associated with the edges of the `graph`, also known as the coupling strengths in spin glasses.
* `vertex_weights` are associated with the vertices of the `graph`, also known the onsite energy term in spin glasses.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of degree of freedoms, where a value can be `0` (in one side of the cut) or `1` (in the other side of the cut).
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.
"""
struct SpinGlass{CT<:AbstractEinsum,WT1<:Union{NoWeight, Vector},WT2<:Union{ZeroWeight, Vector}} <: GraphProblem
    code::CT
    graph::SimpleGraph{Int}
    edge_weights::WT1
    vertex_weights::WT2
    fixedvertices::Dict{Int,Int}
end
function SpinGlass(g::SimpleGraph; edge_weights=NoWeight(), vertex_weights=ZeroWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert edge_weights isa NoWeight || length(edge_weights) == ne(g)
    @assert vertex_weights isa ZeroWeight || length(vertex_weights) == nv(g)
    rawcode = EinCode([
        map(e->[minmax(e.src,e.dst)...], Graphs.edges(g))...,
        map(v->[v], Graphs.vertices(g))...,
    ], collect(Int, openvertices))  # labels for edge tensors
    SpinGlass(_optimize_code(rawcode, uniformsize_fix(rawcode, 2, fixedvertices), optimizer, simplifier), g, edge_weights, vertex_weights, Dict{Int,Int}(fixedvertices))
end

flavors(::Type{<:SpinGlass}) = [0, 1]
# first `ne` indices are for edge weights, last `nv` indices are for vertex weights.
get_weights(gp::SpinGlass, i::Int) = i <= ne(gp.graph) ? [0, gp.edge_weights[i]] : [0, gp.vertex_weights[i-ne(gp.graph)]]
terms(gp::SpinGlass) = getixsv(gp.code)
labels(gp::SpinGlass) = [1:nv(gp.graph)...]
fixedvertices(gp::SpinGlass) = gp.fixedvertices

function generate_tensors(x::T, gp::SpinGlass) where T
    ixs = getixsv(gp.code)
    l = ne(gp.graph)
    tensors = [
        Array{T}[spinglassb((Ref(x) .^ get_weights(gp, i)) ...) for i=1:l]...,
        add_labels!(Array{T}[Ref(x) .^ get_weights(gp, i+l) for i=1:nv(gp.graph)], ixs[l+1:end], labels(gp))...
    ]
    return select_dims(tensors, ixs, fixedvertices(gp))
end

function spinglassb(a, b)
    return [a b; b a]
end

"""
    spinglass_energy(g::SimpleGraph, config; edge_weights=NoWeight(), vertex_weights=ZeroWeight())

Compute the spin glass state energy for the vertex configuration `config` (an iterator).
"""
function spinglass_energy(g::SimpleGraph, config; edge_weights=NoWeight(), vertex_weights=ZeroWeight())
    size = zero(eltype(edge_weights)) * false
    # coupling terms
    for (i, e) in enumerate(edges(g))
        size += (config[e.src] != config[e.dst]) * edge_weights[i]
    end
    # onsite terms
    for (i, v) in enumerate(vertices(g))
        size += config[v] * vertex_weights[i]
    end
    return size
end


"""
    cut_size(g::SimpleGraph, config; weights=NoWeight())

Compute the cut size for the vertex configuration `config` (an iterator).
"""
function cut_size(g::SimpleGraph, config; weights=NoWeight())
    size = zero(eltype(weights)) * false
    for (i, e) in enumerate(edges(g))
        size += (config[e.src] != config[e.dst]) * weights[i]
    end
    return size
end