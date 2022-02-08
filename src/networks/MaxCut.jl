"""
    MaxCut{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
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

Its contraction time space complexity is ``2^{{\\rm tw}(G)}``, where ``{\\rm tw(G)}`` is the [tree-width](https://en.wikipedia.org/wiki/Treewidth) of ``G``.
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

Returns a valid vertex configurations (a vector of size `nv(g)`) from `config` (an iterator) defined on edges:

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