"""
$TYPEDEF

The [binary paint shop problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/PaintShop/).

Positional arguments
-------------------------------
* `sequence` is a vector of symbols, each symbol is associated with a color.
* `isfirst` is a vector of boolean numbers, indicating whether the symbol is the first appearance in the sequence.

Examples
-------------------------------
One can encode the paint shop problem `abaccb` as the following

```jldoctest; setup=:(using GenericTensorNetworks)
julia> syms = collect("abaccb");

julia> pb = GenericTensorNetwork(PaintShop(syms));

julia> solve(pb, SizeMin())[]
2.0ₜ

julia> solve(pb, ConfigsMin())[].c.data
2-element Vector{StaticBitVector{3, 1}}:
 100
 011
```
In our definition, we find the maximum number of unchanged color in this sequence, i.e. (n-1) - (minimum number of color changes)
In the output of maximum configurations, the two configurations are defined on 5 bonds i.e. pairs of (i, i+1), `0` means color changed, while `1` means color not changed.
If we denote two "colors" as `r` and `b`, then the optimal painting is `rbbbrr` or `brrrbb`, both change the colors twice.
"""
energy_terms(gp::PaintShop) = [[gp.sequence[i], gp.sequence[i+1]] for i in 1:length(gp.sequence)-1]
energy_tensors(x::T, c::PaintShop) where T = [flip_labels(paintshop_bond_tensor(_pow.(Ref(x), get_weights(c, i))...), c.isfirst[i], c.isfirst[i+1]) for i=1:length(c.sequence)-1]
extra_terms(::PaintShop{LT}) where LT = Vector{LT}[]
extra_tensors(::Type{T}, ::PaintShop) where T = Array{T}[]
labels(gp::PaintShop) = unique(gp.sequence)

# get_weights interface
get_weights(c::PaintShop) = UnitWeight(length(unique(c.sequence)))
get_weights(::PaintShop, i::Int) = [0, 1]

function paintshop_bond_tensor(a::T, b::T) where T
    m = T[a b; b a]
    return m
end
function flip_labels(m, if1, if2)
    m = if1 ? m : m[[2,1],:]
    m = if2 ? m : m[:,[2,1]]
    return m
end
