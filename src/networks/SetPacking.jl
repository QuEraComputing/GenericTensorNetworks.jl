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

flavors(::Type{<:SetPacking}) = [0, 1]
energy_terms(gp::SetPacking) = [[i] for i=1:length(gp.sets)]
energy_tensors(x::T, c::SetPacking) where T = [misv(_pow.(Ref(x), get_weights(c, i))) for i=1:length(c.sets)]
extra_terms(gp::SetPacking) = [[i,j] for i=1:length(gp.sets),j=1:length(gp.sets) if j>i && !isempty(gp.sets[i] ∩ gp.sets[j])]
extra_tensors(::Type{T}, gp::SetPacking) where T = [misb(T, length(ix)) for ix in extra_terms(gp)]
labels(gp::SetPacking) = [1:length(gp.sets)...]

# weights interface
get_weights(c::SetPacking) = c.weights
get_weights(gp::SetPacking, i::Int) = [0, gp.weights[i]]
chweights(c::SetPacking, weights) = SetPacking(c.sets, weights)

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