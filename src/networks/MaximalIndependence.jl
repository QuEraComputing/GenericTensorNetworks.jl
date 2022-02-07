"""
    MaximalIndependence{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
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
struct MaximalIndependence{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    weights::WT
end

function MaximalIndependence(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == nv(g)
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    MaximalIndependence(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), weights)
end

flavors(::Type{<:MaximalIndependence}) = [0, 1]
symbols(gp::MaximalIndependence) = [i for i in 1:length(getixsv(gp.code))]
get_weights(gp::MaximalIndependence, label) = [0, gp.weights isa UnWeighted ? 1 : gp.weights[findfirst(==(label), symbols(gp))]]

function generate_tensors(fx, mi::MaximalIndependence)
    ixs = getixsv(mi.code)
    isempty(ixs) && return []
    T = eltype(fx(ixs[1][end]))
	return map(ixs) do ix
        neighbortensor(fx(ix[end])..., length(ix))
    end
end
function neighbortensor(a::T, b::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = one(T)
    end
    t[1<<(d-1)+1] = a
    t[1<<(d-1)+1] = b
    return t
end

