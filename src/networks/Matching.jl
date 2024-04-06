"""
$TYPEDEF

The [Vertex matching](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Matching/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the edges of the `graph`.
"""
struct Matching{WT<:Union{NoWeight,Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    weights::WT
    function Matching(g::SimpleGraph, weights::Union{NoWeight, Vector}=NoWeight())
        @assert weights isa NoWeight || length(weights) == ne(g)
        new{typeof(weights)}(g, weights)
    end
end

function GenericTensorNetwork(problem::Matching; openvertices=(), fixedvertices=Dict{Int,Int}())
    edges = [minmax(e.src,e.dst) for e in Graphs.edges(problem.graph)]
    rawcode = EinCode(vcat([[s] for s in edges], # labels for edge tensors
                [[minmax(i,j) for j in neighbors(problem.graph, i)] for i in Graphs.vertices(problem.graph)]),
                collect(Tuple{Int,Int}, openvertices))
    return GenericTensorNetwork(problem, rawcode, Dict{Int,Int}(fixedvertices))
end

flavors(::Type{<:Matching}) = [0, 1]
terms(gp::Matching) = getixsv(gp.code)[1:ne(gp.graph)]
labels(gp::Matching) = getindex.(terms(gp))

# weights interface
get_weights(c::Matching) = c.weights
get_weights(m::Matching, i::Int) = [0, m.weights[i]]
chweights(c::Matching, weights) = Matching(c.code, c.graph, weights, c.fixedvertices)

function generate_tensors(x::T, m::GenericTensorNetwork{<:Matching}) where T
    ne(m.problem.graph) == 0 && return Array{T}[]
    ixs = getixsv(m.code)
    tensors = Array{T}[]
    for i=1:length(ixs)
        ix = ixs[i]
        if i<=ne(m.problem.graph)
            @assert length(ix) == 1
            t = _pow.(Ref(x), get_weights(m, i)) # x is defined on edges.
        else
            t = match_tensor(T, length(ix))
        end
        push!(tensors, t)
    end
    return select_dims(add_labels!(tensors, ixs, labels(m)), ixs, fixedvertices(m))
end

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