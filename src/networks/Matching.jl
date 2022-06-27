"""
    Matching{CT<:AbstractEinsum, WT<:Union{NoWeight,Vector}} <: GraphProblem
    Matching(graph; weights=NoWeight(), openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )

The [Vertex matching](https://psychic-meme-f4d866f8.pages.github.io/dev/generated/Matching.html) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.

Keyword arguments
-------------------------------
* `weights` are associated with the edges of the `graph`.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of degree of freedoms on edges, where a value can be `0` (not matched) or `1` (mathced).
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.
"""
struct Matching{CT<:AbstractEinsum, WT<:Union{NoWeight,Vector}} <: GraphProblem
    code::CT
    graph::SimpleGraph{Int}
    weights::WT
    fixedvertices::Dict{Tuple{Int,Int},Int}
end

function Matching(g::SimpleGraph; weights=NoWeight(), openvertices=(),fixedvertices=Dict{Tuple{Int,Int},Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa NoWeight || length(weights) == ne(g)
    edges = [minmax(e.src,e.dst) for e in Graphs.edges(g)]
    rawcode = EinCode(vcat([[s] for s in edges], # labels for edge tensors
                [[minmax(i,j) for j in neighbors(g, i)] for i in Graphs.vertices(g)]),
                collect(Tuple{Int,Int}, openvertices))
    Matching(_optimize_code(rawcode, uniformsize_fix(rawcode, 2, fixedvertices), optimizer, simplifier), g, weights, Dict{Tuple{Int,Int},Int}(fixedvertices))
end

flavors(::Type{<:Matching}) = [0, 1]
get_weights(m::Matching, i::Int) = [0, m.weights[i]]
terms(gp::Matching) = getixsv(gp.code)[1:ne(gp.graph)]
labels(gp::Matching) = getindex.(terms(gp))
fixedvertices(gp::Matching) = gp.fixedvertices

# weights interface
weights(c::Matching) = c.weights
chweights(c::Matching, weights) = Matching(c.code, c.graph, weights, c.fixedvertices)

function generate_tensors(x::T, m::Matching) where T
    ne(m.graph) == 0 && return Array{T}[]
    ixs = getixsv(m.code)
    tensors = Array{T}[]
    for i=1:length(ixs)
        ix = ixs[i]
        if i<=ne(m.graph)
            @assert length(ix) == 1
            t = Ref(x) .^ get_weights(m, i) # x is defined on edges.
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