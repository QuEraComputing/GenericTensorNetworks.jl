const EinTypes = Union{EinCode,NestedEinsum,SlicedEinsum}

abstract type GraphProblem end

struct UnWeighted end

"""
    Independence{CT<:EinTypes,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    Independence(graph; weights=UnWeighted(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

The [Independent set problem](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)).
In the constructor, `weights` are the weights of vertices.
`openvertices` specifies labels for the output tensor.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

Problem definition
---------------------------
An independent set is defined in the [monadic second order logic](https://digitalcommons.oberlin.edu/cgi/viewcontent.cgi?article=1678&context=honors) as
```math
\\exists x_i,\\ldots,x_M\\left[\\bigwedge_{i\\neq j} (x_i\\neq x_j \\wedge \\neg \\textbf{adj}(x_i, x_j))\\right]
```

Graph polynomial
---------------------------
The graph polynomial defined for the independence problem is known as the independence polynomial.
```math
I(G, x) = \\sum_{k=0}^{\\alpha(G)} a_k x^k,
```
where ``\\alpha(G)`` is the maximum independent set size, 
``a_k`` is the number of independent sets of size ``k`` in graph ``G=(V,E)``.
The total number of independent sets is thus equal to ``I(G, 1)``.

Tensor network
---------------------------
In tensor network representation of the independent set problem,
we map a vertex ``i\\in V`` to a label ``s_i \\in \\{0, 1\\}`` of dimension 2,
where we use 0 (1) to denote a vertex is absent (present) in the set.
For each label ``s_i``, we defined a parametrized rank-one vertex tensor ``W(x_i)`` as
```math
W(x_i)_{s_i} = \\left(\\begin{matrix}
    1 \\\\
    x_i
\\end{matrix}\\right)_{s_i}
```
We use subscripts to index tensor elements, e.g.``W(x_i)_0=1`` is the first element associated
with ``s_i=0`` and ``W(x_i)_1=x_i`` is the second element associated with ``s_i=1``.
Similarly, on each edge ``(u, v)``, we define a matrix ``B`` indexed by ``s_u`` and ``s_v`` as
```math
B_{s_i s_j} = \\left(\\begin{matrix}
    1  & 1\\\\
    1 & 0
\\end{matrix}\\right)_{s_is_j}
```

Its contraction time space complexity is ``2^{{\\rm tw}(G)}``, where ``{\\rm tw}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G``.
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
    MaxCut(graph; weights=UnWeighted(), openvertices=(),
                optimizer=GreedyMethod(), simplifier=nothing)

[Cut](https://en.wikipedia.org/wiki/Maximum_cut) problem (or spin glass problem).
In the constructor, `weights` are the weights of edges.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

Problem definition
---------------------------
In graph theory, a cut is a partition of the vertices of a graph into two disjoint subsets.
A maximum cut is a cut whose size is at least the size of any other cut,
where the size of a cut is the number of edges (or the sum of weights on edges) crossing the cut.

Graph polynomial
---------------------------
The graph polynomial defined for the cut problem is
```math
C(G, x) = \\sum_{k=0}^{\\gamma(G)} c_k x^k,
```
where ``\\alpha(G)`` is the maximum independent set size, 
``c_k/2`` is the number of cuts of size ``k`` in graph ``G=(V,E)``.

Tensor network
---------------------------
For a vertex ``v\\in V``, we define a boolean degree of freedom ``s_v\\in\\{0, 1\\}``.
Then the maximum cut problem can be encoded to tensor networks by mapping an edge ``(i,j)\\in E`` to an edge matrix labelled by ``s_is_j``
```math
B(x_{\\langle i, j\\rangle}) = \\left(\\begin{matrix}
    1 & x_{\\langle i, j\\rangle}\\\\
    x_{\\langle i, j\\rangle} & 1
\\end{matrix}\\right),
```
where variable ``x_{\\langle i, j\\rangle}`` represents a cut on edge ``(i, j)`` or a domain wall of an Ising spin glass.
Similar to other problems, we can define a polynomial about edges variables by setting ``x_{\\langle i, j\\rangle} = x``,
where its k-th coefficient is two times the number of configurations of cut size k.

Its contraction time space complexity is ``2^{{\\rm tw}(G)}``, where ``{\\rm tw}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G``.
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
    MaximalIndependence(graph; weights=UnWeighted(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

[Maximal independent set](https://en.wikipedia.org/wiki/Maximal_independent_set) problem. In the constructor, `weights` are the weights of vertices.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

Problem definition
---------------------------
In graph theory, a maximal independent set is an independent set that is not a subset of any other independent set.
It is different from maximum independent set because it does not require the set to have the max size.

Graph polynomial
---------------------------
The graph polynomial defined for the maximal independent set problem is
```math
I_{\\rm max}(G, x) = \\sum_{k=0}^{\\alpha(G)} b_k x^k,
```
where ``b_k`` is the number of maximal independent sets of size ``k`` in graph ``G=(V, E)``.

Tensor network
---------------------------
For a vertex ``v\\in V``, we define a boolean degree of freedom ``s_v\\in\\{0, 1\\}``.
We defined the restriction on its neighbourhood ``N[v]``:
```math
T(x_v)_{s_1,s_2,\\ldots,s_{|N(v)|},s_v} = \\begin{cases}
    s_vx_v & s_1=s_2=\\ldots=s_{|N(v)|}=0,\\\\
    1-s_v& \\text{otherwise}.\\
\\end{cases}
```
Intuitively, it means if all the neighbourhood vertices are not in ``I_{m}``, i.e., ``s_1=s_2=\\ldots=s_{|N(v)|}=0``, then ``v`` should be in ``I_{m}`` and contribute a factor ``x_{v}``,
otherwise, if any of the neighbourhood vertices is in ``I_{m}``, then ``v`` cannot be in ``I_{m}``.

Its contraction time space complexity is no longer determined by the tree-width of the original graph ``G``.
It is often harder to contract this tensor network than to contract the one for regular independent set problem.
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
The matching polynomial adopts the first definition in [wiki page](https://en.wikipedia.org/wiki/Matching_polynomial)
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
    PaintShop(labels::AbstractVector; openvertices=(),
             optimizer=GreedyMethod(), simplifier=nothing)

The [binary paint shop problem](http://m-hikari.com/ams/ams-2012/ams-93-96-2012/popovAMS93-96-2012-2.pdf).

Example
-----------------------------------------
One can encode the paint shop problem `abaccb` as the following

```jldoctest; setup=:(using GraphTensorNetworks)
julia> symbols = collect("abaccb");

julia> pb = PaintShop(symbols);

julia> solve(pb, SizeMax())[]
3.0ₜ

julia> solve(pb, ConfigsMax())[].c.data
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

# TODOs:
# 1. Dominating set
# \exists x_i,\ldots,x_K \forall y\left[\bigwedge_{i=1}^{K}(y=x_i\wedge \textbf{adj}(y, x_i))\right]
# 2. Polish reading data
#     * consistent configuration assign of max-cut
# 3. Support transverse field in max-cut