"""
    PaintShop{CT<:AbstractEinsum} <: GraphProblem
    PaintShop(labels::AbstractVector; openvertices=(),
             optimizer=GreedyMethod(), simplifier=nothing)

The [binary paint shop problem](http://m-hikari.com/ams/ams-2012/ams-93-96-2012/popovAMS93-96-2012-2.pdf).

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
symbols(gp::PaintShop) = getixsv(gp.code)

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

