"""
    Matching{CT<:AbstractEinsum} <: GraphProblem
    Matching(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

[Vertex matching](https://mathworld.wolfram.com/Matching.html) problem.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

Problem definition
---------------------------
A ``k``-matching in a graph ``G`` is a set of k edges, no two of which have a vertex in common.

Graph polynomial
---------------------------
The matching polynomial adopts the first definition in [wiki page](https://en.wikipedia.org/wiki/Matching_polynomial)
```math
M(G, x) = \\sum\\limits_{k=1}^{|V|/2} c_k x^k,
```
where ``k`` is the number of matches, and coefficients ``c_k`` are the corresponding counting.

Tensor network
---------------------------
We map an edge ``(u, v) \\in E`` to a label ``\\langle u, v\\rangle \\in \\{0, 1\\}`` in a tensor network,
where 1 means two vertices of an edge are matched, 0 means otherwise.
Then we define a tensor of rank ``d(v) = |N(v)|`` on vertex ``v`` such that,
```math
W_{\\langle v, n_1\\rangle, \\langle v, n_2 \\rangle, \\ldots, \\langle v, n_{d(v)}\\rangle} = \\begin{cases}
    1, & \\sum_{i=1}^{d(v)} \\langle v, n_i \\rangle \\leq 1,\\\\
    0, & \\text{otherwise},
\\end{cases}
```
and a tensor of rank 1 on the bond
```math
B_{\\langle v, w\\rangle} = \\begin{cases}
1, & \\langle v, w \\rangle = 0 \\\\
x, & \\langle v, w \\rangle = 1,
\\end{cases}
```
where label ``\\langle v, w \\rangle`` is equivalent to ``\\langle w,v\\rangle``.
"""
struct Matching{CT<:AbstractEinsum} <: GraphProblem
    symbols::Vector{Tuple{Int,Int}}
    code::CT
end

function Matching(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    symbols = [minmax(e.src,e.dst) for e in Graphs.edges(g)]
    rawcode = EinCode(vcat([[s] for s in symbols], # labels for edge tensors
                [[minmax(i,j) for j in neighbors(g, i)] for i in Graphs.vertices(g)]),
                collect(Tuple{Int,Int}, openvertices))
    Matching(symbols, _optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier))
end

flavors(::Type{<:Matching}) = [0, 1]
get_weights(::Matching, sym) = [0, 1]
symbols(m::Matching) = m.symbols

function generate_tensors(fx, m::Matching)
    syms = symbols(m)
    isempty(syms) && return []
    ixs = getixsv(m.code)
    T = eltype(fx(syms[1]))
    n = length(syms)  # number of vertices
    tensors = []
    for i=1:length(ixs)
        if i<=n
            @assert length(ixs[i]) == 1
            t = fx(syms[i]) # fx is defined on edges.
        else
            t = match_tensor(T, length(ixs[i]))
        end
        push!(tensors, t)
    end
    return tensors
end
function match_tensor(::Type{T}, n::Int) where T
    t = zeros(T, fill(2, n)...)
    for ci in CartesianIndices(t)
        if sum(ci.I .- 1) <= 1
            t[ci] = one(T)
        end
    end
    return t
end

