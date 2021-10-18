export Independence, MaximalIndependence, Matching, Coloring, optimize_code, set_packing, MaxCut
const EinTypes = Union{EinCode,NestedEinsum}

abstract type GraphProblem end

"""
    Independence{CT<:EinTypes} <: GraphProblem
    Independence(graph; openvertices=(), optmethod=:tree, kwargs...)

Independent set problem. `kwargs` is forwarded to `optimize_code`.
"""
struct Independence{CT<:EinTypes} <: GraphProblem
    code::CT
end

function Independence(g::SimpleGraph; openvertices=(), optmethod=:tree, kwargs...)
    rawcode = EinCode(([(i,) for i in LightGraphs.vertices(g)]..., # labels for vertex tensors
                    [minmax(e.src,e.dst) for e in LightGraphs.edges(g)]...), openvertices)  # labels for edge tensors
    code = optimize_code(rawcode, Val(optmethod); kwargs...)
    Independence(code)
end

"""
    MaxCut{CT<:EinTypes} <: GraphProblem
    MaxCut(graph; openvertices=(), optmethod=:tree, kwargs...)

Max cut problem (or spin glass problem). `kwargs` is forwarded to `optimize_code`.
"""
struct MaxCut{CT<:EinTypes} <: GraphProblem
    code::CT
end
function MaxCut(g::SimpleGraph; openvertices=(), optmethod=:tree, kwargs...)
    rawcode = EinCode(([minmax(e.src,e.dst) for e in LightGraphs.edges(g)]...,), openvertices)  # labels for edge tensors
    MaxCut(optimize_code(rawcode, Val(optmethod); kwargs...))
end

"""
    MaximalIndependence{CT<:EinTypes} <: GraphProblem
    MaximalIndependence(graph; openvertices=(), optmethod=:tree, kwargs...)

Maximal independent set problem. `kwargs` is forwarded to `optimize_code`.
"""
struct MaximalIndependence{CT<:EinTypes} <: GraphProblem
    code::CT
end

function MaximalIndependence(g::SimpleGraph; openvertices=(), optmethod=:tree, kwargs...)
    rawcode = EinCode(([(LightGraphs.neighbors(g, v)..., v) for v in LightGraphs.vertices(g)]...,), openvertices)
    MaximalIndependence(optimize_code(rawcode, Val(optmethod); kwargs...))
end

"""
    Matching{CT<:EinTypes} <: GraphProblem
    Matching(graph; openvertices=(), optmethod=:tree, kwargs...)

Vertex matching problem. `kwargs` is forwarded to `optimize_code`.
The matching polynomial adopts the first definition in wiki page: https://en.wikipedia.org/wiki/Matching_polynomial
```math
m_G(x) := \\sum_{k\\geq 0}m_kx^k,
```
where `m_k` is the number of k-edge matchings.
"""
struct Matching{CT<:EinTypes} <: GraphProblem
    code::CT
end

function Matching(g::SimpleGraph; openvertices=(), optmethod=:tree, kwargs...)
    rawcode = EinCode(([(minmax(e.src,e.dst),) for e in LightGraphs.edges(g)]..., # labels for edge tensors
                    [([minmax(i,j) for j in neighbors(g, i)]...,) for i in LightGraphs.vertices(g)]...,), openvertices)       # labels for vertex tensors
    Matching(optimize_code(rawcode, Val(optmethod); kwargs...))
end

"""
    Coloring{K,CT<:EinTypes} <: GraphProblem
    Coloring{K}(graph; openvertices=(), optmethod=:tree, kwargs...)

K-Coloring problem. `kwargs` is forwarded to `optimize_code`.
"""
struct Coloring{K,CT<:EinTypes} <: GraphProblem
    code::CT
end
Coloring{K}(code::ET) where {K,ET<:EinTypes} = Coloring{K,ET}(code)
# same network layout as independent set.
Coloring{K}(g::SimpleGraph; openvertices=(), optmethod=:tree, kwargs...) where K = Coloring{K}(Independence(g; openvertices=openvertices, kwargs...).code)

"""
    labels(code)

Return a vector of unique labels in an Einsum token.
"""
function labels(code::EinTypes)
    res = []
    for ix in collect_ixs(code)
        for l in ix
            if l ∉ res
                push!(res, l)
            end
        end
    end
    return res
end

collect_ixs(ne::EinCode) = [collect(ix) for ix in getixs(ne)]
function collect_ixs(ne::NestedEinsum)
    d = OMEinsum.collect_ixs!(ne, Dict{Int,Vector{OMEinsum.labeltype(ne.eins)}}())
    return [d[i] for i=1:length(d)]
end

"""
    optimize_code(code; optmethod=:kahypar, sc_target=17, max_group_size=40, nrepeat=10, imbalances=0.0:0.001:0.8, βs=0.01:0.05:10.0, ntrials=50, niters=1000, sc_weight=2.0, rw_weight=1.0)

Optimize the contraction order.

* `optmethod` can be one of
    * `:kahypar`, the kahypar + greedy approach, takes kwargs [`sc_target`, `max_group_size`, `imbalances`, `nrepeat`].
    Check `optimize_kahypar` method in package `OMEinsumContractionOrders`.
    * `:auto`, also the kahypar + greedy approach, but determines `sc_target` automatically. It is slower!
    * `:greedy`, the greedy approach. Check `optimize_greedy` in package `OMEinsum`.
    * `:tree`, the approach of running simulated annealing on expression tree, takes kwargs [`sc_target`, `sc_weight`, `rw_weight`, `βs`, `ntrials`, `niters`]. Check `optimize_tree` in package `OMEinsumContractionOrders`.
    * `:sa`, the simulated annealing approach, takes kwargs [`rw_weight`, `βs`, `ntrials`, `niters`]. Check `optimize_sa` in package `OMEinsumContractionOrders`.
    * `:raw`, do nothing and return the raw EinCode.
"""
function optimize_code(@nospecialize(code::EinTypes), ::Val{optmethod}; sc_target=17, max_group_size=40, nrepeat=10, imbalances=0.0:0.001:0.8, initializer=:random, βs=0.01:0.05:10.0, ntrials=50, niters=1000, sc_weight=2.0, rw_weight=1.0) where optmethod
    size_dict = Dict([s=>2 for s in labels(code)])
    optcode = if optmethod == :kahypar
        optimize_kahypar(code, size_dict; sc_target=sc_target, max_group_size=max_group_size, imbalances=imbalances, greedy_nrepeat=nrepeat)
    elseif optmethod == :sa
        optimize_sa(code, size_dict; sc_target=sc_target, max_group_size=max_group_size, βs=βs, ntrials=ntrials, niters=niters, initializer=initializer, greedy_nrepeat=nrepeat)
    elseif optmethod == :greedy
        optimize_greedy(code, size_dict; nrepeat=nrepeat)
    elseif optmethod == :tree
        optimize_tree(code, size_dict; sc_target=sc_target, βs=βs, niters=niters, ntrials=ntrials, sc_weight=sc_weight, initializer=initializer, rw_weight=rw_weight)
    elseif optmethod == :auto
        optimize_kahypar_auto(code, size_dict; max_group_size=max_group_size, effort=500, greedy_nrepeat=nrepeat)
    elseif optmethod == :raw
        code
    else
        ArgumentError("optimizer `$optmethod` not defined.")
    end
    @info "time/space complexity is $(OMEinsum.timespace_complexity(optcode, size_dict))"
    return optcode
end

OMEinsum.timespace_complexity(gp::GraphProblem) = timespace_complexity(gp.code, uniformsize(gp.code, bondsize(gp)))

for T in [:Independence, :Matching, :MaximalIndependence, :MaxCut]
    @eval bondsize(gp::$T) = 2
end
bondsize(gp::Coloring{K}) where K = K

"""
set_packing(sets; openvertices=(), optmethod=:tree, kwargs...)

Set packing is a generalization of independent set problem to hypergraphs.
Calling this function will return you an `Independence` instance.
`sets` are a vector of vectors, each element being a vertex in the independent set problem.
`kwargs` is forwarded to `optimize_code`.

### Example
```julia
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = set_packing(sets; optmethod=:auto);

julia> res = best_solutions(gp; all=true)[]
(2, {10010, 00110, 01100})ₜ
```
"""
function set_packing(sets; openvertices=(), optmethod=:tree, kwargs...)
    n = length(sets)
    code = EinCode(([(i,) for i=1:n]..., [(i,j) for i=1:n,j=1:n if j>i && !isempty(sets[i] ∩ sets[j])]...), ())
    Independence(optimize_code(code, Val(optmethod); kwargs...))
end
