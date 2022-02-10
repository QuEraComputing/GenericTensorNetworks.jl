"""
    MaximalIS{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    MaximalIS(graph; weights=UnWeighted(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

The [maximal independent set](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/MaximalIS.html) problem. In the constructor, `weights` are the weights of vertices.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct MaximalIS{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    weights::WT
end

function MaximalIS(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == nv(g)
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    MaximalIS(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), weights)
end

flavors(::Type{<:MaximalIS}) = [0, 1]
symbols(gp::MaximalIS) = [i for i in 1:length(getixsv(gp.code))]
get_weights(gp::MaximalIS, label) = [0, gp.weights[findfirst(==(label), symbols(gp))]]

function generate_tensors(fx, mi::MaximalIS)
    ixs = getixsv(mi.code)
    isempty(ixs) && return []
    T = eltype(fx(ixs[1][end]))
	return map(ixs) do ix
        neighbortensor(fx(ix[end])..., length(ix))
    end
end
function neighbortensor(a::T, b::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = one(T)
    end
    t[1<<(d-1)+1] = a
    t[1<<(d-1)+1] = b
    return t
end

