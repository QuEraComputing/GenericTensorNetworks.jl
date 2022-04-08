"""
    SetCover{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    SetCover(sets; weights=NoWeight(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

The [set cover problem](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/SetCover.html).

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
```julia
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = SetCover(sets);

julia> res = best_solutions(gp; all=true)[]
(2, {10010, 00110, 01100})ₜ
```
"""
struct SetCover{ET, CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    sets::Vector{Vector{ET}}
    weights::WT
end

function SetCover(sets; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where ET
    nsets = length(sets)
    # get constraints
    elements, count = cover_count(sets)

    code = EinCode([[[i] for i=1:nsets]...,
        [count[e] for e in elements]...], collect(Int,openvertices))
    SetCover(_optimize_code(code, uniformsize(code, 2), optimizer, simplifier), sets, weights)
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

flavors(::Type{<:SetCover}) = [0, 1]
get_weights(gp::SetCover, i::Int) = [0, gp.weights[i]]
terms(gp::SetCover) = getixsv(gp.code)[1:length(gp.sets)]
labels(gp::SetCover) = [1:length(gp.sets)...]

# generate tensors
function generate_tensors(x::T, gp::SetCover) where T
    length(gp.sets) == 0 && return []
    ixs = getixsv(gp.code)
    # we only add labels at vertex tensors
    return vcat(add_labels!([misv(Ref(x) .^ get_weights(gp, i)) for i=1:length(gp.sets)], ixs[1:length(gp.sets)], labels(gp)),
            [cover_tensor(T, length(ix)) for ix in ixs[length(gp.sets)+1:end]]
    )
end

"""
    is_set_cover(sets::AbstractVector, config)

Return true if `config` (a vector of boolean numbers as the mask of sets) is a set cover of `sets`.
"""
function is_set_cover(sets::AbstractVector{ST}, config) where ST
    count = cover_count(sets[(!iszero).(config)])
    return all(isone, values(d))
end