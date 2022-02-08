"""
    Coloring{K,CT<:AbstractEinsum} <: GraphProblem
    Coloring{K}(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

[Vertex Coloring](https://en.wikipedia.org/wiki/Graph_coloring) problem.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

Problem definition
---------------------------
A vertex coloring is an assignment of labels or colors to each vertex of a graph such that no edge connects two identically colored vertices. 

Tensor network
---------------------------
Let us use 3-colouring problem defined on vertices as an example.
For a vertex ``v``, we define the degree of freedoms ``c_v\\in\\{1,2,3\\}`` and a vertex tensor labelled by it as
```math
W(v) = \\left(\\begin{matrix}
    r_v\\\\
    g_v\\\\
    b_v
\\end{matrix}\\right).
```
For an edge ``(u, v)``, we define an edge tensor as a matrix labelled by ``(c_u, c_v)`` to specify the constraint
```math
B = \\left(\\begin{matrix}
    0 & 1 & 1\\\\
    1 & 0 & 1\\\\
    1 & 1 & 0
\\end{matrix}\\right).
```
The number of possible colouring can be obtained by contracting this tensor network by setting vertex tensor elements ``r_v, g_v`` and ``b_v`` to 1.
"""
struct Coloring{K,CT<:AbstractEinsum} <: GraphProblem
    code::CT
    nv::Int
end
Coloring{K}(code::ET, nv::Int) where {K,ET<:AbstractEinsum} = Coloring{K,ET}(code, nv)
# same network layout as independent set.
Coloring{K}(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where K = Coloring{K}(Independence(g; openvertices=openvertices, optimizer=optimizer, simplifier=simplifier).code, nv(g))

flavors(::Type{<:Coloring{K}}) where K = collect(0:K-1)
symbols(c::Coloring{K}) where K = [i for i=1:c.nv]

# `fx` is a function defined on symbols, it returns a vector of elements, the size of vector is same as the number of flavors (or the bond dimension).
function generate_tensors(fx, c::Coloring{K}) where K
    ixs = getixsv(c.code)
    T = eltype(fx(ixs[1][1]))
    return map(1:length(ixs)) do i
        i <= c.nv ? coloringv(fx(ixs[i][1])) : coloringb(T, K)
    end
end

# coloring bond tensor
function coloringb(::Type{T}, k::Int) where T
    x = ones(T, k, k)
    for i=1:k
        x[i,i] = zero(T)
    end
    return x
end
# coloring vertex tensor
coloringv(vals::Vector{T}) where T = vals

