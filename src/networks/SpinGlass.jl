"""
    SpinGlass{TT<:MaxCut,WT1<:Union{NoWeight, Vector},WT2<:Union{ZeroWeight, Vector}} <: ReducedProblem
    SpinGlass(graph; J=NoWeight(), h=ZeroWeight(), openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )
    SpinGlass(M::AbstractMatrix, h::AbstractVector; kwargs...)

The [spin glass](https://psychic-meme-f4d866f8.pages.github.io/dev/generated/SpinGlass.html) problem (or cutting problem).
In the output, the spin ↑ is mapped to configuration 0, while spin ↓ is mapped to configuration 1.

Positional arguments
-------------------------------
* `graph` is the problem graph.

Keyword arguments
-------------------------------
* `M` is a symmetric matrix of the coupling strengths.
* `J` is a vector of coupling strengths associated with the edges of the `graph`.
* `h` is a vector of onsite energy terms associated with the vertices of the `graph`.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of degree of freedoms, where a value can be `0` (in one side of the cut) or `1` (in the other side of the cut).
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.
"""
struct SpinGlass{TT<:MaxCut,WT1<:Union{NoWeight, Vector},WT2<:Union{ZeroWeight, Vector}} <: ReducedProblem
    target::TT
    J::WT1
    h::WT2
end

function SpinGlass(g::SimpleGraph; J=NoWeight(), h=ZeroWeight(), kwargs...)
    @assert J isa NoWeight || length(J) == ne(g)
    @assert h isa ZeroWeight || length(h) == nv(g)
    SpinGlass(MaxCut(g; edge_weights=[2*J[i] for i=1:ne(g)], vertex_weights=[-2*h[i] for i=1:nv(g)], kwargs...), J, h)
end

function SpinGlass(M::AbstractMatrix, h::AbstractVector; kwargs...)
    g = SimpleGraph((!iszero).(M))
    J = [M[e.src, e.dst] for e in edges(g)]
    return SpinGlass(g; J, h, kwargs...)
end

target_problem(sg::SpinGlass) = sg.target

# the energy should be shifted by sum(J)/2 - sum(h)
for ET in [:Tropical, :ExtendedTropical, :TruncatedPoly, :CountingTropical]
    @eval function extract_result(sg::SpinGlass, res::T) where T <: $(ET)
        sumJ = sum(i->sg.J[i], 1:ne(sg.target.graph))
        sumh = sum(i->sg.h[i], 1:nv(sg.target.graph))
        # the cut size is always even if the input J is integer
        return res * _x(T; invert=false) ^ (-sumJ - sumh)
    end
end
function extract_result(sg::SpinGlass, res::Union{Polynomial{BS,X}, LaurentPolynomial{BS,X}}) where {BS,X}
    sumJ = sum(i->sg.J[i], 1:ne(sg.target.graph))
    sumh = sum(i->sg.h[i], 1:nv(sg.target.graph))
    # the cut size is always even if the input J is integer
    lres = LaurentPolynomial{BS,X}(res)
    return lres * LaurentPolynomial{BS,X}([one(eltype(res.coeffs))], -sumJ + sumh)
end

# the configurations are not changed
for ET in [:SumProductTree, :ConfigSampler, :ConfigEnumerator, :Real]
    @eval extract_result(sg::SpinGlass, res::T) where T <: $(ET) = res
end

"""
    spinglass_energy(g::SimpleGraph, config; J=NoWeight(), h=ZeroWeight())

Compute the spin glass state energy for the vertex configuration `config`.
In the configuration, the spin ↑ is mapped to configuration 0, while spin ↓ is mapped to configuration 1.
Let ``G=(V,E)`` be the input graph, the hamiltonian is
```math
H = - \\sum_{ij \\in E} J_{ij} s_i s_j + \\sum_{i \\in V} h_i s_i,
```
where ``s_i \\in \\{-1, 1\\}`` stands for spin ↓ and spin ↑.
"""
function spinglass_energy(g::SimpleGraph, config; J=NoWeight(), h=ZeroWeight())
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