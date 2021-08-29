export Independence, MaximalIndependence, Matching, Coloring, optimize_code, set_packing, SpinGlass
const EinTypes = Union{EinCode,NestedEinsum}

abstract type GraphProblem end

"""
    Independence{CT<:EinTypes} <: GraphProblem
    Independence(graph; kwargs...)

Independent set problem. For `kwargs`, check `optimize_code` API.
"""
struct Independence{CT<:EinTypes} <: GraphProblem
    code::CT
end

function Independence(g::SimpleGraph; outputs=(), kwargs...)
    rawcode = EinCode(([(i,) for i in LightGraphs.vertices(g)]..., # labels for vertex tensors
                    [minmax(e.src,e.dst) for e in LightGraphs.edges(g)]...), outputs)  # labels for edge tensors
    Independence(optimize_code(rawcode; kwargs...))
end

"""
    SpinGlass{Q, CT<:EinTypes} <: GraphProblem
    SpinGlass{Q}(graph; kwargs...)
    SpinGlass(graph; kwargs...)

Q-state spin glass problem (or Potts model). For `kwargs`, check `optimize_code` API.
When Q=2, it corresponds to the {0, 1} spin glass model.
"""
struct SpinGlass{Q, CT<:EinTypes} <: GraphProblem
    code::CT
end

SpinGlass(g::SimpleGraph; outputs=(), kwargs...) = SpinGlass{2}(g; outputs=outputs, kwargs...)
SpinGlass{Q}(g::SimpleGraph; outputs=(), kwargs...) where Q = SpinGlass{Q}(Independence(g; outputs=outputs, kwargs...).code)
SpinGlass{Q}(code::EinTypes) where Q = SpinGlass{Q,typeof(code)}(code)

"""
    MaximalIndependence{CT<:EinTypes} <: GraphProblem
    MaximalIndependence(graph; kwargs...)

Maximal independent set problem. For `kwargs`, check `optimize_code` API.
"""
struct MaximalIndependence{CT<:EinTypes} <: GraphProblem
    code::CT
end

function MaximalIndependence(g::SimpleGraph; outputs=(), kwargs...)
    rawcode = EinCode(([(LightGraphs.neighbors(g, v)..., v) for v in LightGraphs.vertices(g)]...,), outputs)
    MaximalIndependence(optimize_code(rawcode; kwargs...))
end

"""
    Matching{CT<:EinTypes} <: GraphProblem
    Matching(graph; kwargs...)

Vertex matching problem. For `kwargs`, check `optimize_code` API.
"""
struct Matching{CT<:EinTypes} <: GraphProblem
    code::CT
end

function Matching(g::SimpleGraph; outputs=(), kwargs...)
    rawcode = EinCode(([(minmax(e.src,e.dst),) for e in LightGraphs.edges(g)]..., # labels for edge tensors
                    [([minmax(i,j) for j in neighbors(g, i)]...,) for i in LightGraphs.vertices(g)]...,), outputs)       # labels for vertex tensors
    Matching(optimize_code(rawcode; kwargs...))
end

"""
    Coloring{K,CT<:EinTypes} <: GraphProblem
    Coloring{K}(graph; kwargs...)

K-Coloring problem. For `kwargs`, check `optimize_code` API.
"""
struct Coloring{K,CT<:EinTypes} <: GraphProblem
    code::CT
end
Coloring{K}(code::ET) where {K,ET<:EinTypes} = Coloring{K,ET}(code)
# same network layout as independent set.
Coloring(g::SimpleGraph; outputs=(), kwargs...) = Coloring(Independence(g; outputs=outputs, kwargs...).code)

"""
    labels(code)

Return a vector of unique labels in an Einsum token.
"""
function labels(code::EinTypes)
    res = []
    for ix in OMEinsum.getixs(OMEinsum.flatten(code))
        for l in ix
            if l ∉ res
                push!(res, l)
            end
        end
    end
    return res
end

"""
    optimize_code(code; optmethod=:kahypar, sc_target=17, max_group_size=40, nrepeat=10, imbalances=0.0:0.001:0.8)

Optimize the contraction order.

* `optmethod` can be one of
    * `:kahypar`, the kahypar + greedy approach, takes kwargs [`sc_target`, `max_group_size`, `imbalances`].
    Check `optimize_kahypar` method in package `OMEinsumContractionOrders`.
    * `:auto`, also the kahypar + greedy approach, but determines `sc_target` automatically. It is slower!
    * `:greedy`, the greedy approach. Check in `optimize_greedy` in package `OMEinsum`.
    * `:raw`, do nothing and return the raw EinCode.
"""
function optimize_code(code::EinTypes; optmethod=:auto, sc_target=17, max_group_size=40, nrepeat=10, imbalances=0.0:0.001:0.8)
    size_dict = Dict([s=>2 for s in labels(code)])
    optcode = if optmethod == :kahypar
        optimize_kahypar(code, size_dict; sc_target=sc_target, max_group_size=max_group_size, imbalances=imbalances)
    elseif optmethod == :greedy
        optimize_greedy(code, size_dict; nrepeat=nrepeat)
    elseif optmethod == :auto
        optimize_kahypar_auto(code, size_dict; max_group_size=max_group_size, effort=500)
    elseif optmethod == :raw
        code
    else
        ArgumentError("optimizer `$optmethod` not defined.")
    end
    println("time/space complexity is $(OMEinsum.timespace_complexity(optcode, size_dict))")
    return optcode
end

OMEinsum.timespace_complexity(gp::GraphProblem) = timespace_complexity(gp.code, uniformsize(gp.code, bondsize(gp)))

for T in [:Independence, :Matching, :MaximalIndependence]
    @eval bondsize(gp::$T) = 2
end
bondsize(gp::Coloring{K}) where K = K
bondsize(gp::SpinGlass{Q}) where Q = Q

"""
    set_packing(sets; kwargs...)

Set packing is a generalization of independent set problem to hypergraphs.
Calling this function will return you an `Independence` instance.
`sets` are a vector of vectors, each element being a vertex in the independent set problem. `kwargs` are code optimization parameters.

### Example
```julia
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = set_packing(sets; optmethod=:auto);

julia> res = optimalsolutions(gp; all=true)[]
(2, {10010, 00110, 01100})ₜ
```
"""
function set_packing(sets; kwargs...)
    n = length(sets)
    code = EinCode(([(i,) for i=1:n]..., [(i,j) for i=1:n,j=1:n if j>i && !isempty(sets[i] ∩ sets[j])]...), ())
    Independence(optimize_code(code; kwargs...))
end