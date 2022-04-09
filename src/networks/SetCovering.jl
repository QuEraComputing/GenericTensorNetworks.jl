"""
    SetCovering{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    SetCovering(sets; weights=NoWeight(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

The [set covering problem](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/SetCovering.html).

Positional arguments
-------------------------------
* `sets` is a vector of vectors, each set is associated with a weight specified in `weights`.

Keyword arguments
-------------------------------
* `weights` are associated with sets.
* `openvertices` specifies labels of the output tensor.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.

Example
-----------------------------------
```jldoctest; setup=:(using GraphTensorNetworks)
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = SetCovering(sets);

julia> res = solve(gp, ConfigsMin())[]
(3.0, {10110, 10101})ₜ
```
"""
struct SetCovering{ET, CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    sets::Vector{Vector{ET}}
    weights::WT
end

function SetCovering(sets; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where ET
    nsets = length(sets)
    @assert weights isa NoWeight || length(weights) == nsets
    # get constraints
    elements, count = cover_count(sets)

    code = EinCode([[[i] for i=1:nsets]...,
        [count[e] for e in elements]...], collect(Int,openvertices))
    SetCovering(_optimize_code(code, uniformsize(code, 2), optimizer, simplifier), sets, weights)
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

function cover_tensor(::Type{T}, set_indices::AbstractVector{Int}) where T
    n = length(set_indices)
    t = ones(T, fill(2, n)...)
    t[1] = zero(T)
    return t
end

flavors(::Type{<:SetCovering}) = [0, 1]
get_weights(gp::SetCovering, i::Int) = [0, gp.weights[i]]
terms(gp::SetCovering) = getixsv(gp.code)[1:length(gp.sets)]
labels(gp::SetCovering) = [1:length(gp.sets)...]

# generate tensors
function generate_tensors(x::T, gp::SetCovering) where T
    nsets = length(gp.sets)
    nsets == 0 && return []
    ixs = getixsv(gp.code)
    # we only add labels at vertex tensors
    return vcat(add_labels!([misv(Ref(x) .^ get_weights(gp, i)) for i=1:nsets], ixs[1:nsets], labels(gp)),
            [cover_tensor(T, ix) for ix in ixs[nsets+1:end]]
    )
end

"""
    is_set_covering(sets::AbstractVector, config)

Return true if `config` (a vector of boolean numbers as the mask of sets) is a set covering of `sets`.
"""
function is_set_covering(sets::AbstractVector, config)
    insets = sets[(!iszero).(config)]
    return length(unique!(vcat(insets...))) == length(unique!(vcat(sets...)))
end