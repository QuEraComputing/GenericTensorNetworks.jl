"""
    PaintShop{CT<:AbstractEinsum} <: GraphProblem
    PaintShop(sequence::AbstractVector; openvertices=(),
            optimizer=GreedyMethod(), simplifier=nothing,
            fixedvertices=Dict()
        )

The [binary paint shop problem](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/PaintShop.html).

Positional arguments
-------------------------------

Keyword arguments
-------------------------------
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of degree of freedoms, where a value can be `0` (the first appearence in blue) or `1` (the first appearence in red).
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.

Examples
-------------------------------
One can encode the paint shop problem `abaccb` as the following

```jldoctest; setup=:(using GenericTensorNetworks)
julia> syms = collect("abaccb");

julia> pb = PaintShop(syms);

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
struct PaintShop{CT<:AbstractEinsum,LT} <: GraphProblem
    code::CT
    sequence::Vector{LT}
    isfirst::Vector{Bool}
    fixedvertices::Dict{LT,Int}
end

function paintshop_from_pairs(pairs::AbstractVector{Tuple{Int,Int}}; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing, fixedvertices=Dict())
    n = length(pairs)
    @assert sort!(vcat(collect.(pairs)...)) == collect(1:2n)
    sequence = zeros(Int, 2*n)
    @inbounds for i=1:n
        sequence[pairs[i]] .= i
    end
    return PaintShop(pairs; openvertices, optimizer, simplifier, fixedvertices)
end
function PaintShop(sequence::AbstractVector{T}; openvertices=(), fixedvertices=Dict{T,Int}(), optimizer=GreedyMethod(), simplifier=nothing) where T
    @assert all(l->count(==(l), sequence)==2, sequence)
    n = length(sequence)
    isfirst = [findfirst(==(sequence[i]), sequence) == i for i=1:n]
    rawcode = EinCode(vcat(
                [[sequence[i], sequence[i+1]] for i=1:n-1], # labels for edge tensors
                ),
                collect(T, openvertices))
    PaintShop(_optimize_code(rawcode, uniformsize_fix(rawcode, 2, fixedvertices), optimizer, simplifier), sequence, isfirst, Dict{LT,Int}(fixedvertices))
end

flavors(::Type{<:PaintShop}) = [0, 1]
get_weights(::PaintShop, i::Int) = [0, 1]
terms(gp::PaintShop) = getixsv(gp.code)
labels(gp::PaintShop) = unique(gp.sequence)
fixedvertices(gp::PaintShop) = gp.fixedvertices

function generate_tensors(x::T, c::PaintShop) where T
    ixs = getixsv(c.code)
    tensors = [paintshop_bond_tensor((Ref(x) .^ get_weights(c, i))...) for i=1:length(ixs)]
    return select_dims(add_labels!([flip_labels(tensors[i], c.isfirst[i], c.isfirst[i+1]) for i=1:length(ixs)], ixs, labels(c)), ixs, fixedvertices(c))
end

function paintshop_bond_tensor(a::T, b::T) where T
    m = T[a b; b a]
    return m
end
function flip_labels(m, if1, if2)
    m = if1 ? m : m[[2,1],:]
    m = if2 ? m : m[:,[2,1]]
    return m
end

"""
    num_paint_shop_color_switch(sequence::AbstractVector, coloring)

Returns the number of color switches.
"""
function num_paint_shop_color_switch(sequence::AbstractVector, coloring)
    return count(i->coloring[i] != coloring[i+1], 1:length(sequence)-1)
end

"""
    paint_shop_coloring_from_config(p::PaintShop, config)

Returns a valid painting from the paint shop configuration (given by the configuration solvers).
The `config` is a sequence of 0 and 1, where 0 means painting the first appearence of a car in blue, 1 otherwise.
"""
function paint_shop_coloring_from_config(p::PaintShop{CT,LT}, config) where {CT, LT}
    d = Dict{LT,Bool}(zip(labels(p), config))
    return map(1:length(p.sequence)) do i
        p.isfirst[i] ? d[p.sequence[i]] : ~d[p.sequence[i]]
    end
end
