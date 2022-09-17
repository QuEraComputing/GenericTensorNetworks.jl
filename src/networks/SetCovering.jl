"""
    SetCovering{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    SetCovering(sets; weights=NoWeight(), openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )

The [set covering problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/SetCovering/).

Positional arguments
-------------------------------
* `sets` is a vector of vectors, each set is associated with a weight specified in `weights`.

Keyword arguments
-------------------------------
* `weights` are associated with sets.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of degree of freedoms, where a value can be `0` (absent in the set) or `1` (present in the set).
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.

Examples
-------------------------------
```jldoctest; setup=:(using GenericTensorNetworks)
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
    fixedvertices::Dict{ET,Int}
end

function SetCovering(sets::AbstractVector{Vector{ET}}; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing, fixedvertices=Dict{ET,Int}()) where ET
    nsets = length(sets)
    @assert weights isa NoWeight || length(weights) == nsets
    # get constraints
    elements, count = cover_count(sets)

    code = EinCode([[[i] for i=1:nsets]...,
        [count[e] for e in elements]...], collect(Int,openvertices))
    SetCovering(_optimize_code(code, uniformsize_fix(code, 2, fixedvertices), optimizer, simplifier), sets, weights, Dict{ET,Int}(fixedvertices))
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
terms(gp::SetCovering) = getixsv(gp.code)[1:length(gp.sets)]
labels(gp::SetCovering) = [1:length(gp.sets)...]
fixedvertices(gp::SetCovering) = gp.fixedvertices

# weights interface
get_weights(c::SetCovering) = c.weights
get_weights(gp::SetCovering, i::Int) = [0, gp.weights[i]]
chweights(c::SetCovering, weights) = SetCovering(c.code, c.sets, weights, c.fixedvertices)

# generate tensors
function generate_tensors(x::T, gp::SetCovering) where T
    nsets = length(gp.sets)
    nsets == 0 && return []
    ixs = getixsv(gp.code)
    # we only add labels at vertex tensors
    return select_dims([
        add_labels!(Array{T}[misv(_pow.(Ref(x), get_weights(gp, i))) for i=1:nsets], ixs[1:nsets], labels(gp))...,
            Array{T}[cover_tensor(T, ix) for ix in ixs[nsets+1:end]]...], ixs, fixedvertices(gp)
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