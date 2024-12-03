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

julia> gp = GenericTensorNetwork(SetPacking(sets));

julia> res = solve(gp, ConfigsMax())[]
(2.0, {00110, 10010, 01100})ₜ
```
"""
energy_terms(gp::SetPacking) = [[i] for i=1:length(gp.sets)]
energy_tensors(x::T, c::SetPacking) where T = [misv(_pow.(Ref(x), get_weights(c, i))) for i=1:length(c.sets)]
extra_terms(gp::SetPacking) = [[i,j] for i=1:length(gp.sets),j=1:length(gp.sets) if j>i && !isempty(gp.sets[i] ∩ gp.sets[j])]
extra_tensors(::Type{T}, gp::SetPacking) where T = [misb(T, length(ix)) for ix in extra_terms(gp)]
labels(gp::SetPacking) = [1:length(gp.sets)...]
# weights interface
get_weights(c::SetPacking) = c.weights
get_weights(gp::SetPacking, i::Int) = [0, gp.weights[i]]