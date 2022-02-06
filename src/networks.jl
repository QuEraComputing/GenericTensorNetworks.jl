export Independence, MaximalIndependence, Matching, Coloring, optimize_code, set_packing, MaxCut, PaintShop, paintshop_from_pairs, UnWeighted
const EinTypes = Union{EinCode,NestedEinsum,SlicedEinsum}

abstract type GraphProblem end

struct UnWeighted end

"""
    Independence{CT<:EinTypes,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    Independence(graph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Independent set problem. In the constructor, `weights` are the weights of vertices.
`openvertices` specifies labels for the output tensor.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct Independence{CT<:EinTypes,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    weights::WT
end

function Independence(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == nv(g)
    rawcode = EinCode(([[i] for i in Graphs.vertices(g)]..., # labels for vertex tensors
                       [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]...), collect(Int, openvertices))  # labels for edge tensors
    code = _optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier)
    Independence(code, weights)
end

"""
    MaxCut{CT<:EinTypes,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    MaxCut(graph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Max cut problem (or spin glass problem). In the constructor, `weights` are the weights of edges.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct MaxCut{CT<:EinTypes,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    weights::WT
end
function MaxCut(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == ne(g)
    rawcode = EinCode([[minmax(e.src,e.dst)...] for e in Graphs.edges(g)], collect(Int, openvertices))  # labels for edge tensors
    MaxCut(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), weights)
end

"""
    MaximalIndependence{CT<:EinTypes,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    MaximalIndependence(graph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Maximal independent set problem. In the constructor, `weights` are the weights of vertices.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct MaximalIndependence{CT<:EinTypes,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    weights::WT
end

function MaximalIndependence(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == nv(g)
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    MaximalIndependence(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), weights)
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
    PaintShop{CT<:EinTypes} <: GraphProblem
    PaintShop(labels::AbstractVector; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

The binary paint shop problem: http://m-hikari.com/ams/ams-2012/ams-93-96-2012/popovAMS93-96-2012-2.pdf.

Example
-----------------------------------------
One can encode the paint shop problem `abaccb` as the following

```jldoctest; setup=:(using GraphTensorNetworks)
julia> symbols = collect("abaccb");

julia> pb = PaintShop(symbols);

julia> solve(pb, "size max")[]
3.0ₜ

julia> solve(pb, "configs max")[].c.data
2-element Vector{StaticBitVector{5, 1}}:
 01101
 01101
```
In our definition, we find the maximum number of unchanged color in this sequence, i.e. (n-1) - (minimum number of color changes)
In the output of maximum configurations, the two configurations are defined on 5 bonds i.e. pairs of (i, i+1), `0` means color changed, while `1` means color not changed.
If we denote two "colors" as `r` and `b`, then the optimal painting is `rbbbrr` or `brrrbb`, both change the colors twice.
"""
struct PaintShop{CT<:EinTypes,LT} <: GraphProblem
    code::CT
    labels::Vector{LT}
    isfirst::Vector{Bool}
end

function paintshop_from_pairs(pairs::AbstractVector{Tuple{Int,Int}}; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    n = length(pairs)
    @assert sort!(vcat(collect.(pairs)...)) == collect(1:2n)
    labels = zeros(Int, 2*n)
    @inbounds for i=1:n
        labels[pairs[i]] .= i
    end
    return PaintShop(pairs; openvertices, optimizer, simplifier)
end
function PaintShop(labels::AbstractVector{T}; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where T
    @assert all(l->count(==(l), labels)==2, labels)
    n = length(labels)
    isfirst = [findfirst(==(labels[i]), labels) == i for i=1:n]
    rawcode = EinCode(vcat(
                [[labels[i], labels[i+1]] for i=1:n-1], # labels for edge tensors
                ),
                collect(T, openvertices))
    PaintShop(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), labels, isfirst)
end

"""
    labels(code)

Return a vector of unique labels in an Einsum token.
"""
function labels(code::EinTypes)
    res = []
    for ix in getixsv(code)
        for l in ix
            if l ∉ res
                push!(res, l)
            end
        end
    end
    return res
end

OMEinsum.timespace_complexity(gp::GraphProblem) = timespace_complexity(gp.code, uniformsize(gp.code, bondsize(gp)))

for T in [:Independence, :Matching, :MaximalIndependence, :MaxCut, :PaintShop]
    @eval bondsize(gp::$T) = 2
end
bondsize(::Coloring{K}) where K = K

get_weight(gp::GraphProblem, x::Int) = 1
for T in [:Independence, :MaximalIndependence, :MaxCut]
    @eval get_weight(gp::$T, x::Int) = gp.weights isa UnWeighted ? 1 : gp.weights[x]
end

"""
set_packing(sets; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Set packing is a generalization of independent set problem to hypergraphs.
Calling this function will return you an `Independence` instance.
`sets` are a vector of vectors, each element being a vertex in the independent set problem.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

Example
-----------------------------------
```julia
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = set_packing(sets);

julia> res = best_solutions(gp; all=true)[]
(2, {10010, 00110, 01100})ₜ
```
"""
function set_packing(sets; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    n = length(sets)
    code = EinCode(vcat([[i] for i=1:n], [[i,j] for i=1:n,j=1:n if j>i && !isempty(sets[i] ∩ sets[j])]), collect(Int,openvertices))
    Independence(_optimize_code(code, uniformsize(code, 2), optimizer, simplifier), weights)
end

_optimize_code(code, size_dict, optimizer::Nothing, simplifier) = code
_optimize_code(code, size_dict, optimizer, simplifier) = optimize_code(code, size_dict, optimizer, simplifier)

# TODO:
# 1. Dominating set
# \exists x_i,\ldots,x_K \forall y\left[\bigwedge_{i=1}^{K}(y=x_i\wedge \textbf{adj}(y, x_i))\right]
# 2. Polish reading data
#     * consistent configuration assign of max-cut
# 3. Support transverse field in max-cut