export Independence, MaximalIndependence, Matching, Coloring, optimize_code, set_packing, MaxCut
const EinTypes = Union{EinCode,NestedEinsum,SlicedEinsum}

abstract type GraphProblem end

"""
    Independence{CT<:EinTypes} <: GraphProblem
    Independence(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Independent set problem. `openvertices` specifies the output tensor.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct Independence{CT<:EinTypes} <: GraphProblem
    code::CT
end

function Independence(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    rawcode = EinCode(([[i] for i in Graphs.vertices(g)]..., # labels for vertex tensors
                       [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]...), collect(Int, openvertices))  # labels for edge tensors
    code = _optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier)
    Independence(code)
end

"""
    MaxCut{CT<:EinTypes} <: GraphProblem
    MaxCut(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Max cut problem (or spin glass problem).
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct MaxCut{CT<:EinTypes} <: GraphProblem
    code::CT
end
function MaxCut(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    rawcode = EinCode([[minmax(e.src,e.dst)...] for e in Graphs.edges(g)], collect(Int, openvertices))  # labels for edge tensors
    MaxCut(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier))
end

"""
    MaximalIndependence{CT<:EinTypes} <: GraphProblem
    MaximalIndependence(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Maximal independent set problem. 
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct MaximalIndependence{CT<:EinTypes} <: GraphProblem
    code::CT
end

function MaximalIndependence(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    MaximalIndependence(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier))
end

"""
    Matching{CT<:EinTypes} <: GraphProblem
    Matching(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Vertex matching problem.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
The matching polynomial adopts the first definition in wiki page: https://en.wikipedia.org/wiki/Matching_polynomial
```math
m_G(x) := \\sum_{k\\geq 0}m_kx^k,
```
where `m_k` is the number of k-edge matchings.
"""
struct Matching{CT<:EinTypes} <: GraphProblem
    code::CT
end

function Matching(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    rawcode = EinCode(vcat([[minmax(e.src,e.dst)] for e in Graphs.edges(g)], # labels for edge tensors
                [[minmax(i,j) for j in neighbors(g, i)] for i in Graphs.vertices(g)]),
                collect(Tuple{Int,Int}, openvertices))
    Matching(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier))
end

"""
    Coloring{K,CT<:EinTypes} <: GraphProblem
    Coloring{K}(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

K-Coloring problem.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct Coloring{K,CT<:EinTypes} <: GraphProblem
    code::CT
end
Coloring{K}(code::ET) where {K,ET<:EinTypes} = Coloring{K,ET}(code)
# same network layout as independent set.
Coloring{K}(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where K = Coloring{K}(Independence(g; openvertices=openvertices, optimizer=optimizer, simplifier=simplifier).code)

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

OMEinsum.timespace_complexity(gp::GraphProblem) = timespace_complexity(gp.code, uniformsize(gp.code, bondsize(gp)))

for T in [:Independence, :Matching, :MaximalIndependence, :MaxCut]
    @eval bondsize(gp::$T) = 2
end
bondsize(gp::Coloring{K}) where K = K

"""
set_packing(sets; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Set packing is a generalization of independent set problem to hypergraphs.
Calling this function will return you an `Independence` instance.
`sets` are a vector of vectors, each element being a vertex in the independent set problem.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

### Example
```julia
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = set_packing(sets);

julia> res = best_solutions(gp; all=true)[]
(2, {10010, 00110, 01100})ₜ
```
"""
function set_packing(sets; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    n = length(sets)
    code = EinCode(vcat([[i] for i=1:n], [[i,j] for i=1:n,j=1:n if j>i && !isempty(sets[i] ∩ sets[j])]), collect(Int,openvertices))
    Independence(_optimize_code(code, uniformsize(code, 2), optimizer, simplifier))
end

_optimize_code(code, size_dict, optimizer::Nothing, simplifier) = code
_optimize_code(code, size_dict, optimizer, simplifier) = optimize_code(code, size_dict, optimizer, simplifier)
