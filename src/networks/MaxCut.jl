"""
    MaxCut{CT<:AbstractEinsum,WT<:Union{NoWieght, Vector}} <: GraphProblem
    MaxCut(graph; weights=NoWeight(), openvertices=(),
                optimizer=GreedyMethod(), simplifier=nothing)

The [cutting](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/MaxCut.html) problem (or spin glass problem).
In the constructor, `weights` are the weights of edges.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct MaxCut{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    graph::SimpleGraph{Int}
    weights::WT
    fix_config
end
function MaxCut(g::SimpleGraph; weights=NoWeight(), openvertices=(), fix_config=Dict(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa NoWeight || length(weights) == ne(g)
    rawcode = EinCode([[minmax(e.src,e.dst)...] for e in Graphs.edges(g)], collect(Int, openvertices))  # labels for edge tensors
    size_dict = uniformsize(rawcode, 2)
    for key in keys(fix_config)
        size_dict[key] = 1
    end
    MaxCut(_optimize_code(rawcode, size_dict, optimizer, simplifier), g, weights, fix_config)
end

flavors(::Type{<:MaxCut}) = [0, 1]
get_weights(gp::MaxCut, i::Int) = [0, gp.weights[i]]
terms(gp::MaxCut) = getixsv(gp.code)
labels(gp::MaxCut) = [1:nv(gp.graph)...]
select_dims(gp::MaxCut, ix) = (ix[1] ∉ keys(gp.fix_config) ? (1:2) : (gp.fix_config[ix[1]]+1:gp.fix_config[ix[1]]+1), ix[2] ∉ keys(gp.fix_config) ? (1:2) : (gp.fix_config[ix[2]]+1:gp.fix_config[ix[2]]+1))

function generate_tensors(x::T, gp::MaxCut) where T
    ixs = getixsv(gp.code)
    tensors = map(enumerate(ixs)) do (i, ix)
        maxcutb((Ref(x) .^ get_weights(gp, i), select_dims(gp, ix)) ...)
    end
    return add_labels!(tensors, ixs, labels(gp), gp.fix_config)
end

function maxcutb(x, select_dims=(1:2,1:2))
    a, b = x
    m = [a b; b a]
    m = m[select_dims[1], select_dims[2]]
    return m
end

"""
    cut_size(g::SimpleGraph, config; weights=NoWeight())

Compute the cut size from vertex `config` (an iterator).
"""
function cut_size(g::SimpleGraph, config; weights=NoWeight())
    size = zero(eltype(weights)) * false
    for (i, e) in enumerate(edges(g))
        size += (config[e.src] != config[e.dst]) * weights[i]
    end
    return size
end