"""
$TYPEDEF

The [spin glass](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/SpinGlass/) problem (or cutting problem).
In the output, the spin ↑ is mapped to configuration 0, while spin ↓ is mapped to configuration 1.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `J` is a vector of coupling strengths associated with the edges of the `graph`.
* `h` is a vector of onsite energy terms associated with the vertices of the `graph`.
"""
struct SpinGlass{WT1<:Union{UnitWeight, Vector}, WT2<:Union{ZeroWeight, Vector}} <: GraphProblem
    graph::SimpleGraph{Int}
    J::WT1
    h::WT2
    function SpinGlass(g::SimpleGraph, J::WT1=UnitWeight(), h::WT2=ZeroWeight()) where {WT1, WT2}
        @assert J isa UnitWeight || length(J) == ne(g)
        @assert h isa ZeroWeight || length(h) == nv(g)
        new{WT1, WT2}(g, J, h)
    end
end
function spin_glass_from_matrix(M::AbstractMatrix, h::AbstractVector)
    g = SimpleGraph((!iszero).(M))
    J = [M[e.src, e.dst] for e in edges(g)]
    return SpinGlass(g, J, h)
end

function GenericTensorNetwork(cfg::SpinGlass; openvertices=(), fixedvertices=Dict{Int,Int}())
    g, J, h = cfg.graph, cfg.J, cfg.h
    reducedto = MaxCut(g, edge_weights=[2*J[i] for i=1:ne(g)], vertex_weights=[-2*h[i] for i=1:nv(g)])
    return GenericTensorNetwork(reducedto; openvertices, fixedvertices)
end

# the energy should be shifted by sum(J)/2 - sum(h)
function extract_result(sg::SpinGlass)
    sumJ = sum(i->sg.J[i], 1:ne(sg.target.graph))
    sumh = sum(i->sg.h[i], 1:nv(sg.target.graph))
    function extractor(res::T) where T <: Union{Tropical, ExtendedTropical, TruncatedPoly, CountingTropical}
        # the cut size is always even if the input J is integer
        return res * _x(T; invert=false) ^ (-sumJ + sumh)
    end
    function extractor(res::T) where {BS, X, T <: Union{Polynomial{BS,X}, LaurentPolynomial{BS,X}}}
        lres = LaurentPolynomial{BS,X}(res)
        return lres * LaurentPolynomial{BS,X}([one(eltype(res.coeffs))], -sumJ + sumh)
    end
    function extractor(res::T) where T<:Union{SumProductTree, ConfigSampler, ConfigEnumerator, Real, AbstractVector}
        return res
    end
    return extractor
end

"""
    spinglass_energy(g::SimpleGraph, config; J, h=ZeroWeight())

Compute the spin glass state energy for the vertex configuration `config`.
In the configuration, the spin ↑ is mapped to configuration 0, while spin ↓ is mapped to configuration 1.
Let ``G=(V,E)`` be the input graph, the hamiltonian is
```math
H = - \\sum_{ij \\in E} J_{ij} s_i s_j + \\sum_{i \\in V} h_i s_i,
```
where ``s_i \\in \\{-1, 1\\}`` stands for spin ↓ and spin ↑.
"""
function spinglass_energy(g::SimpleGraph, config; J, h=ZeroWeight())
    eng = zero(promote_type(eltype(J), eltype(h)))
    # NOTE: cast to Int to avoid using unsigned :nt
    s = 1 .- 2 .* Int.(config)  # 0 -> spin 1, 1 -> spin -1
    # coupling terms
    for (i, e) in enumerate(edges(g))
        eng += (s[e.src] * s[e.dst]) * -J[i]
    end
    # onsite terms
    for (i, v) in enumerate(vertices(g))
        eng += s[v] * h[i]
    end
    return eng
end

# function solve(gp::ReducedProblem, property::AbstractProperty; T=Float64, usecuda=false)
#     res = solve(target_problem(gp), property; T, usecuda)
#     return asarray(extract_result(gp).(res), res)
# end

