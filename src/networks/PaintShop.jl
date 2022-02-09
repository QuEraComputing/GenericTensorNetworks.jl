"""
    PaintShop{CT<:AbstractEinsum} <: GraphProblem
    PaintShop(labels::AbstractVector; openvertices=(),
             optimizer=GreedyMethod(), simplifier=nothing)

The [binary paint shop problem](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/PaintShop.html).

Example
-----------------------------------------
One can encode the paint shop problem `abaccb` as the following

```jldoctest; setup=:(using GraphTensorNetworks)
julia> symbols = collect("abaccb");

julia> pb = PaintShop(symbols);

julia> solve(pb, SizeMax())[]
3.0ₜ

julia> solve(pb, ConfigsMax())[].c.data
2-element Vector{StaticBitVector{5, 1}}:
 01101
 01101
```
In our definition, we find the maximum number of unchanged color in this sequence, i.e. (n-1) - (minimum number of color changes)
In the output of maximum configurations, the two configurations are defined on 5 bonds i.e. pairs of (i, i+1), `0` means color changed, while `1` means color not changed.
If we denote two "colors" as `r` and `b`, then the optimal painting is `rbbbrr` or `brrrbb`, both change the colors twice.
"""
struct PaintShop{CT<:AbstractEinsum,LT} <: GraphProblem
    code::CT
    labels::Vector{LT}
    isfirst::Vector{Bool}
end

function paintshop_from_pairs(pairs::AbstractVector{Tuple{Int,Int}}; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    n = length(pairs)
    @assert sort!(vcat(collect.(pairs)...)) == collect(1:2n)
    labels = zeros(Int, 2*n)
    @inbounds for i=1:n
        labels[pairs[i]] .= i
    end
    return PaintShop(pairs; openvertices, optimizer, simplifier)
end
function PaintShop(labels::AbstractVector{T}; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where T
    @assert all(l->count(==(l), labels)==2, labels)
    n = length(labels)
    isfirst = [findfirst(==(labels[i]), labels) == i for i=1:n]
    rawcode = EinCode(vcat(
                [[labels[i], labels[i+1]] for i=1:n-1], # labels for edge tensors
                ),
                collect(T, openvertices))
    PaintShop(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), labels, isfirst)
end

flavors(::Type{<:PaintShop}) = [0, 1]
get_weights(::PaintShop, sym) = [0, 1]
symbols(gp::PaintShop) = getixsv(gp.code)  # !!! may not be unique

function generate_tensors(fx, c::PaintShop)
    ixs = getixsv(c.code)
    [paintshop_bond_tensor(fx(ixs[i])..., c.isfirst[i], c.isfirst[i+1]) for i=1:length(ixs)]
end

function paintshop_bond_tensor(a::T, b::T, if1::Bool, if2::Bool) where T
    m = T[b a; a b]
    m = if1 ? m : m[[2,1],:]
    m = if2 ? m : m[:,[2,1]]
    return m
end

"""
    num_paint_shop_color_switch(labels::AbstractVector, coloring::AbstractVector)

Check the validity of the `coloring` and returns the number of color switches.
"""
function num_paint_shop_color_switch(labels::AbstractVector, coloring::AbstractVector)
    # check validity of solution
    @assert length(unique(coloring)) == 2 && length(labels) == length(coloring)
    unique_labels = unique(labels)
    @show coloring
    for l in unique_labels
        locs = findall(==(l), labels)
        @assert length(locs) == 2
        c1, c2 = coloring[locs]
        @show c1, c2
        #@assert c1 != c2
    end
    # counting color switch
    return count(i->coloring[i] != coloring[i+1], 1:length(coloring)-1)
end

"""
    paint_shop_coloring_from_config(config)

Return a valid painting from the paint shop configuration (given by the configuration solvers).
The `config` is a sequence of 0 and 1, where 0 means the color changed, 1 mean color unchanged.
"""
function paint_shop_coloring_from_config(config)
    res = falses(length(config)+1)
    @inbounds for i=2:length(res)
        res[i] = res[i-1] ⊻ (1-config[i-1])
    end
    return res
end

function fx_solutions(gp::PaintShop, ::Type{BT}, all::Bool) where {BT}
    syms = symbols(gp)
    T = (all ? set_type : sampler_type)(BT, length(syms), nflavor(gp))
    counter = Ref(0)
    return function (l)
        _onehotv.(Ref(T), (counter[]+=1; counter[]), flavors(gp), get_weights(gp, l))
    end
end