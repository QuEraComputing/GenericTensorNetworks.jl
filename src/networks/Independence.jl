"""
    Independence{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    Independence(graph; weights=UnWeighted(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

The [Independent set problem](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)).
In the constructor, `weights` are the weights of vertices.
`openvertices` specifies labels for the output tensor.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

Problem definition
---------------------------
In graph theory, an independent set is a set of vertices in a graph, no two of which are adjacent.

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

Its contraction time space complexity is ``2^{{\\rm tw}(G)}``, where ``{\\rm tw(G)}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G``.
"""
struct Independence{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    nv::Int
    weights::WT
end

function Independence(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == nv(g)
    rawcode = EinCode(([[i] for i in Graphs.vertices(g)]..., # labels for vertex tensors
                       [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]...), collect(Int, openvertices))  # labels for edge tensors
    code = _optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier)
    Independence(code, nv(g), weights)
end

flavors(::Type{<:Independence}) = [0, 1]
symbols(gp::Independence) = [i for i in 1:gp.nv]
get_weights(gp::Independence, label) = [0, gp.weights isa UnWeighted ? 1 : gp.weights[findfirst(==(label), symbols(gp))]]

# generate tensors
function generate_tensors(fx, gp::Independence)
    syms = symbols(gp)
    isempty(syms) && return []
    ixs = getixsv(gp.code)
    T = eltype(fx(syms[1]))
    return map(enumerate(ixs)) do (i, ix)
        if i <= length(syms)
            misv(fx(ix[1]))
        else
            misb(T, length(ix)) # if n!=2, it corresponds to set packing problem.
        end
    end
end

function misb(::Type{T}, n::Integer=2) where T
    res = zeros(T, fill(2, n)...)
    res[1] = one(T)
    for i=1:n
        res[1+1<<(i-1)] = one(T)
    end
    return res
end
misv(vals) = vals

############### set packing #####################
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
    Independence(_optimize_code(code, uniformsize(code, 2), optimizer, simplifier), n, weights)
end

