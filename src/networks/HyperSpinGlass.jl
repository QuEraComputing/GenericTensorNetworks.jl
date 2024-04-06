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

function hyper_spin_glass_network(n::Int, cliques; weights=UnitWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=MergeVectors())
    cfg = HyperSpinGlass(n, cliques, weights)
    gtn = GenericTensorNetwork(cfg; openvertices, fixedvertices)
    return OMEinsum.optimize_code(gtn; optimizer, simplifier)
end

flavors(::Type{<:HyperSpinGlass}) = [0, 1]
# first `ne` indices are for edge weights, last `n` indices are for vertex weights.
energy_terms(gp::HyperSpinGlass) = gp.cliques
energy_tensors(x::T, c::HyperSpinGlass) where T = [clique_tensor(length(c.cliques[i]), _pow.(Ref(x), get_weights(c, i))...) for i=1:length(c.cliques)]
extra_terms(::HyperSpinGlass) = [[i] for i=1:problem.n]
extra_tensors(::Type{T}, c::HyperSpinGlass) where T = [Array{T}[one(T), one(T)] for i=1:c.n]
labels(gp::HyperSpinGlass) = collect(1:gp.n)

# weights interface
get_weights(c::HyperSpinGlass) = c.weights
get_weights(gp::HyperSpinGlass, i::Int) = [-gp.weights[i], gp.weights[i]]
chweights(c::HyperSpinGlass, weights) = HyperSpinGlass(c.n, c.cliques, weights)

function generate_tensors(x::T, gp::GenericTensorNetwork{<:HyperSpinGlass}) where T
    ixs = getixsv(gp.code)
    l = length(gp.problem.cliques)
    tensors = [
               Array{T}[clique_tensor(length(gp.cliques[i]), _pow.(Ref(x), get_weights(gp, i))...) for i=1:l]...,
               add_labels!(Array{T}[[one(T), one(T)] for i in labels(gp)], ixs[l+1:end], labels(gp))...
    ]
    return select_dims(tensors, ixs, fixedvertices(gp))
end

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
