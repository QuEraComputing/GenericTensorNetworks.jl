"""
    SetPacking{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    SetPacking(sets; weights=NoWeight(), openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )

The [set packing problem](https://psychic-meme-f4d866f8.pages.github.io/dev/generated/SetPacking.html), a generalization of independent set problem to hypergraphs.

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
```jldoctest; setup=:(using GenericTensorNetworks, Random; Random.seed!(2))
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = SetPacking(sets);

julia> res = solve(gp, ConfigsMax())[]
(2.0, {00110, 10010, 01100})ₜ
```
"""
struct SetPacking{ET, CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    sets::Vector{Vector{ET}}
    weights::WT
    fixedvertices::Dict{ET,Int}
end

function SetPacking(sets::AbstractVector{Vector{ET}}; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing, fixedvertices=Dict{ET,Int}()) where ET
    nsets = length(sets)
    @assert weights isa NoWeight || length(weights) == nsets
    code = EinCode(vcat([[i] for i=1:nsets], [[i,j] for i=1:nsets,j=1:nsets if j>i && !isempty(sets[i] ∩ sets[j])]), collect(Int,openvertices))
    SetPacking(_optimize_code(code, uniformsize_fix(code, 2, openvertices), optimizer, simplifier), sets, weights, Dict{ET,Int}(fixedvertices))
end

flavors(::Type{<:SetPacking}) = [0, 1]
get_weights(gp::SetPacking, i::Int) = [0, gp.weights[i]]
terms(gp::SetPacking) = getixsv(gp.code)[1:length(gp.sets)]
labels(gp::SetPacking) = [1:length(gp.sets)...]
fixedvertices(gp::SetPacking) = gp.fixedvertices

# generate tensors
function generate_tensors(x::T, gp::SetPacking) where T
    length(gp.sets) == 0 && return []
    ixs = getixsv(gp.code)
    # we only add labels at vertex tensors
    return select_dims([
        add_labels!(Array{T}[misv(Ref(x) .^ get_weights(gp, i)) for i=1:length(gp.sets)], ixs[1:length(gp.sets)], labels(gp))...,
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