"""
$TYPEDEF

The [set covering problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/SetCovering/).

Positional arguments
-------------------------------
* `sets` is a vector of vectors, each set is associated with a weight specified in `weights`.
* `weights` are associated with sets.

Examples
-------------------------------
```jldoctest; setup=:(using GenericTensorNetworks)
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = GenericTensorNetwork(SetCovering(sets));

julia> res = solve(gp, ConfigsMin())[]
(3.0, {10110, 10101})ₜ
```
"""
struct SetCovering{ET, WT<:Union{UnitWeight, Vector}} <: GraphProblem
    sets::Vector{Vector{ET}}
    weights::WT
    function SetCovering(sets::Vector{Vector{ET}}, weights::Union{UnitWeight, Vector}=UnitWeight()) where {ET}
        @assert weights isa UnitWeight || length(weights) == length(sets)
        new{ET, typeof(weights)}(sets, weights)
    end
end

flavors(::Type{<:SetCovering}) = [0, 1]
energy_terms(gp::SetCovering) = [[i] for i=1:length(gp.sets)]
energy_tensors(x::T, c::SetCovering) where T = [misv(_pow.(Ref(x), get_weights(c, i))) for i=1:length(c.sets)]
function extra_terms(sc::SetCovering)
    elements, count = cover_count(sc.sets)
    return [count[e] for e in elements]
end
extra_tensors(::Type{T}, cfg::SetCovering) where T = [cover_tensor(T, ix) for ix in extra_terms(cfg)]
labels(gp::SetCovering) = [1:length(gp.sets)...]

# weights interface
get_weights(c::SetCovering) = c.weights
get_weights(gp::SetCovering, i::Int) = [0, gp.weights[i]]
chweights(c::SetCovering, weights) = SetCovering(c.sets, weights)

function cover_tensor(::Type{T}, set_indices::AbstractVector{Int}) where T
    n = length(set_indices)
    t = ones(T, fill(2, n)...)
    t[1] = zero(T)
    return t
end

function cover_count(sets)
    elements = vcat(sets...)
    count = Dict{eltype(elements), Vector{Int}}()
    for (iset, set) in enumerate(sets)
        for e in set
            if haskey(count, e)
                push!(count[e], iset)
            else
                count[e] = [iset]
            end
        end
    end
    return elements, count
end

"""
    is_set_covering(sets::AbstractVector, config)

Return true if `config` (a vector of boolean numbers as the mask of sets) is a set covering of `sets`.
"""
function is_set_covering(sets::AbstractVector, config)
    insets = sets[(!iszero).(config)]
    return length(unique!(vcat(insets...))) == length(unique!(vcat(sets...)))
end