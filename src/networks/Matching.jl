"""
$TYPEDEF

The [Vertex matching](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Matching/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the edges of the `graph`.
"""
struct Matching{WT<:Union{UnitWeight,Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    weights::WT
    function Matching(g::SimpleGraph, weights::Union{UnitWeight, Vector}=UnitWeight())
        @assert weights isa UnitWeight || length(weights) == ne(g)
        new{typeof(weights)}(g, weights)
    end
end
function matching_network(g::SimpleGraph; weights=UnitWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=MergeVectors())
    cfg = Matching(g, weights)
    gtn = GenericTensorNetwork(cfg; openvertices, fixedvertices)
    return OMEinsum.optimize_code(gtn; optimizer, simplifier)
end

flavors(::Type{<:Matching}) = [0, 1]
energy_terms(gp::Matching) = [[minmax(src(s), dst(s))] for s in edges(gp.graph)] # edge tensors
energy_tensors(x::T, c::Matching) where T = [_pow.(Ref(x), get_weights(c, i)) for i=1:ne(c.graph)]
extra_terms(gp::Matching) = [[minmax(i, j) for j in neighbors(gp.graph, i)] for i in Graphs.vertices(gp.graph)]
extra_tensors(::Type{T}, gp::Matching) where T = [match_tensor(T, degree(gp.graph, i)) for i=1:nv(gp.graph)]
labels(gp::Matching) = getindex.(energy_terms(gp))

# weights interface
get_weights(c::Matching) = c.weights
get_weights(m::Matching, i::Int) = [0, m.weights[i]]
chweights(c::Matching, weights) = Matching(c.graph, weights)

function match_tensor(::Type{T}, n::Int) where T
    t = zeros(T, fill(2, n)...)
    for ci in CartesianIndices(t)
        if sum(ci.I .- 1) <= 1
            t[ci] = one(T)
        end
    end
    return t
end

"""
    is_matching(graph::SimpleGraph, config)

Returns true if `config` is a valid matching on `graph`, and `false` if a vertex is double matched.
`config` is a vector of boolean variables, which has one to one correspondence with `edges(graph)`.
"""
function is_matching(g::SimpleGraph, config)
    @assert ne(g) == length(config)
    edges_mask = zeros(Bool, nv(g))
    for (e, c) in zip(edges(g), config)
        if !iszero(c)
            if edges_mask[e.src]
                @debug "Vertex $(e.src) is double matched!"
                return false
            end
            if edges_mask[e.dst]
                @debug "Vertex $(e.dst) is double matched!"
                return false
            end
            edges_mask[e.src] = true
            edges_mask[e.dst] = true
        end
    end
    return true
end