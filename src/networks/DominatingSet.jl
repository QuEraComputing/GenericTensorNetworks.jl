"""
$TYPEDEF
    DominatingSet(graph; weights=UnitWeight())

The [dominating set](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/DominatingSet/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the vertices of the `graph`, default to `UnitWeight()`.
"""
struct DominatingSet{WT<:Union{UnitWeight, Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    weights::WT
    function DominatingSet(g::SimpleGraph, weights::Union{UnitWeight, Vector}=UnitWeight())
        @assert weights isa UnitWeight || length(weights) == nv(g)
        new{typeof(weights)}(g, weights)
    end
end
function dominating_set_network(g::SimpleGraph; weights=UnitWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=MergeVectors())
    cfg = DominatingSet(g, weights)
    gtn = GenericTensorNetwork(cfg; openvertices, fixedvertices)
    return OMEinsum.optimize_code(gtn; optimizer, simplifier)
end

flavors(::Type{<:DominatingSet}) = [0, 1]
energy_terms(gp::DominatingSet) = [[Graphs.neighbors(gp.graph, v)..., v] for v in Graphs.vertices(gp.graph)]
energy_tensors(x::T, c::DominatingSet) where T = [dominating_set_tensor(_pow.(Ref(x), get_weights(c, i))..., degree(c.graph, i)+1) for i=1:nv(c.graph)]
extra_terms(::DominatingSet) = Vector{Int}[]
extra_tensors(::Type{T}, ::DominatingSet) where T = Array{T}[]
labels(gp::DominatingSet) = [1:nv(gp.graph)...]

# weights interface
get_weights(c::DominatingSet) = c.weights
get_weights(gp::DominatingSet, i::Int) = [0, gp.weights[i]]
chweights(c::DominatingSet, weights) = DominatingSet(c.graph, weights)

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