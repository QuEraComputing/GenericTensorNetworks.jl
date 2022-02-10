"""
    MaxCut{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    MaxCut(graph; weights=UnWeighted(), openvertices=(),
                optimizer=GreedyMethod(), simplifier=nothing)

[Cut](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/MaxCut.html) problem (or spin glass problem).
In the constructor, `weights` are the weights of edges.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct MaxCut{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    weights::WT
end
function MaxCut(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == ne(g)
    rawcode = EinCode([[minmax(e.src,e.dst)...] for e in Graphs.edges(g)], collect(Int, openvertices))  # labels for edge tensors
    MaxCut(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), weights)
end

flavors(::Type{<:MaxCut}) = [0, 1]
symbols(gp::MaxCut) = getixsv(gp.code)
get_weights(gp::MaxCut, label) = [0, gp.weights[findfirst(==(label), symbols(gp))]]

function generate_tensors(fx, gp::MaxCut)
    ixs = getixsv(gp.code)
    return map(enumerate(ixs)) do (i, ix)
        maxcutb(fx(ix)...)
    end
end
function maxcutb(a, b)
    return [a b; b a]
end

"""
    cut_assign(g::SimpleGraph, config)

Returns a valid vertex cut configurations (a vector of size `nv(g)`) from `config` (an iterator) defined on edges:

* assign two vertices with the same values if they are connected by an edge with configuration `0`.
* assign two vertices with the different values if they are connected by an edge with configuration `1`.
* error if there is no valid assignment.
"""
function cut_assign(g::SimpleGraph, config)
    nv(g) == 0 && return Bool[]
    assign = fill(-1, nv(g))
    @inbounds assign[1] = 0
    nassign = 1
    @inbounds while nassign != nv(g)
        for (e, c) in zip(edges(g), config)
            if assign[e.src] == -1 && assign[e.dst] != -1
                assign[e.src] = assign[e.dst] ⊻ c
            elseif assign[e.src] != -1 && assign[e.dst] == -1
                assign[e.dst] = assign[e.src] ⊻ c
            elseif assign[e.src] != -1 && assign[e.dst] != -1
                if assign[e.dst] != assign[e.src] ⊻ c
                    error("Assign conflict on edge $e, current assign is $(assign[e.src]) and $(assign[e.dst]).")
                end
            end
        end
        _nassign = count(!=(-1), assign)
        if nassign == _nassign
            assign[findfirst(==(-1), assign)] = 0
            _nassign += 1
        end
        nassign = _nassign
    end
    return assign
end

"""
    cut_size(g::SimpleGraph, config; weights=UnWeighted())

Compute the cut size from vertex `config` (an iterator).
"""
function cut_size(g::SimpleGraph, config; weights=UnWeighted())
    size = zero(eltype(weights)) * false
    for (i, e) in enumerate(edges(g))
        size += (config[e.src] != config[e.dst]) * weights[i]
    end
    return size
end