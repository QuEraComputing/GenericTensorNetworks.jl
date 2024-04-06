"""
$TYPEDEF

The [set packing problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/SetPacking/), a generalization of independent set problem to hypergraphs.

Positional arguments
-------------------------------
* `sets` is a vector of vectors, each set is associated with a weight specified in `weights`.
* `weights` are associated with sets.

Examples
-------------------------------
```jldoctest; setup=:(using GenericTensorNetworks, Random; Random.seed!(2))
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = SetPacking(sets);

julia> res = solve(gp, ConfigsMax())[]
(2.0, {00110, 10010, 01100})ₜ
```
"""
struct SetPacking{ET,WT<:Union{UnitWeight, Vector}} <: GraphProblem
    sets::Vector{Vector{ET}}
    weights::WT
    function SetPacking(sets::Vector{Vector{ET}}, weights::Union{UnitWeight, Vector}=UnitWeight()) where {ET}
        @assert weights isa UnitWeight || length(weights) == length(sets)
        new{ET, typeof(weights)}(sets, weights)
    end
end
function set_packing_network(sets; weights=UnitWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=MergeVectors())
    cfg = SetPacking(sets, weights)
    gtn = GenericTensorNetwork(cfg; openvertices, fixedvertices)
    return OMEinsum.optimize_code(gtn; optimizer, simplifier)
end

flavors(::Type{<:SetPacking}) = [0, 1]
energy_terms(gp::SetPacking) = [[i] for i=1:length(gp.sets)]
extra_terms(gp::SetPacking) = [[i,j] for i=1:length(gp.sets),j=1:length(gp.sets) if j>i && !isempty(gp.sets[i] ∩ gp.sets[j])]
labels(gp::SetPacking) = [1:length(gp.sets)...]

# weights interface
get_weights(c::SetPacking) = c.weights
get_weights(gp::SetPacking, i::Int) = [0, gp.weights[i]]
chweights(c::SetPacking, weights) = SetPacking(c.sets, weights)

# generate tensors
function generate_tensors(x::T, gp::GenericTensorNetwork{<:SetPacking}) where T
    length(gp.problem.sets) == 0 && return []
    ixs = getixsv(gp.code)
    # we only add labels at vertex tensors
    return select_dims([
        add_labels!(Array{T}[misv(_pow.(Ref(x), get_weights(gp, i))) for i=1:length(gp.problem.sets)], ixs[1:length(gp.problem.sets)], labels(gp))...,
        Array{T}[misb(T, length(ix)) for ix in ixs[length(gp.sets)+1:end]]...], ixs, fixedvertices(gp),
    )
end

"""
    is_set_packing(sets::AbstractVector, config)

Return true if `config` (a vector of boolean numbers as the mask of sets) is a set packing of `sets`.
"""
function is_set_packing(sets::AbstractVector{ST}, config) where ST
    d = Dict{eltype(ST), Int}()
    for i=1:length(sets)
        if !iszero(config[i])
            for e in sets[i]
                d[e] = get(d, e, 0) + 1
            end
        end
    end
    return all(isone, values(d))
end