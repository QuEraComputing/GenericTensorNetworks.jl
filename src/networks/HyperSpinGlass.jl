"""
$(TYPEDEF)

The [hyper-spin-glass](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/HyperSpinGlass/) problem is a generalization of the spin-glass problem to hypergraphs.

Positional arguments
-------------------------------
* `n` is the number of spins.
* `cliques` is a vector of cliques, each being a vector of vertices (integers).
* `weights` are associated with the cliques, default to `UnitWeight()`.
"""
struct HyperSpinGlass{WT<:Union{UnitWeight, Vector}} <: GraphProblem
    n::Int
    cliques::Vector{Vector{Int}}
    weights::WT
    function HyperSpinGlass(n::Int, cliques::AbstractVector, weights::Union{UnitWeight, Vector}=UnitWeight())
        clqs = collect(collect.(cliques))
        @assert weights isa UnitWeight || length(weights) == length(clqs)
        @assert all(c->all(b->1<=b<=n, c), cliques) "vertex index out of bound 1-$n, got: $cliques"
        return new{typeof(weights)}(n, clqs, weights)
    end
end
function HyperSpinGlass(graph::SimpleGraph, J::Union{UnitWeight, Vector}, h::Union{ZeroWeight, Vector})
    J_ = J isa UnitWeight ? fill(1, ne(graph)) : J
    h_ = h isa ZeroWeight ? fill(0, nv(graph)) : h
    @assert length(J_) == ne(graph) "length of J must be equal to the number of edges $(ne(graph)), got: $(length(J_))"
    @assert length(h_) == nv(graph) "length of h must be equal to the number of vertices $(nv(graph)), got: $(length(h_))"
    HyperSpinGlass(nv(graph), [[[src(e), dst(e)] for e in edges(graph)]..., [[i] for i in 1:nv(graph)]...], [J_..., h_...])
end
function spin_glass_from_matrix(M::AbstractMatrix, h::AbstractVector)
    g = SimpleGraph((!iszero).(M))
    J = [M[e.src, e.dst] for e in edges(g)]
    return HyperSpinGlass(g, J, h)
end

flavors(::Type{<:HyperSpinGlass}) = [0, 1]
# first `ne` indices are for edge weights, last `n` indices are for vertex weights.
energy_terms(gp::HyperSpinGlass) = gp.cliques
energy_tensors(x::T, c::HyperSpinGlass) where T = [clique_tensor(length(c.cliques[i]), _pow.(Ref(x), get_weights(c, i))...) for i=1:length(c.cliques)]
extra_terms(sg::HyperSpinGlass) = [[i] for i=1:sg.n]
extra_tensors(::Type{T}, c::HyperSpinGlass) where T = [[one(T), one(T)] for i=1:c.n]
labels(gp::HyperSpinGlass) = collect(1:gp.n)

# weights interface
get_weights(c::HyperSpinGlass) = c.weights
get_weights(gp::HyperSpinGlass, i::Int) = [-gp.weights[i], gp.weights[i]]
chweights(c::HyperSpinGlass, weights) = HyperSpinGlass(c.n, c.cliques, weights)

function clique_tensor(rank, a::T, b::T) where T
    res = zeros(T, fill(2, rank)...)
    for i=0:(1<<rank-1)
        res[i+1] = (count_ones(i) % 2) == 1 ? a : b
    end
    return res
end

"""
$(TYPEDSIGNATURES)

Compute the energy for spin configuration `config` (an iterator).
"""
function hyperspinglass_energy(cliques, config; weights=UnitWeight())::Real
    size = zero(eltype(weights))
    s = 1 .- 2 .* Int.(config)  # 0 -> spin 1, 1 -> spin -1
    for (i, spins) in enumerate(cliques)
        size += prod(s[spins]) * weights[i]
    end
    return size
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
        eng += (s[e.src] * s[e.dst]) * J[i]
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
