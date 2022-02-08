"""
    MaximalIndependentSet{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    MaximalIndependentSet(graph; weights=UnWeighted(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

[Maximal independent set](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/MaximalIndependentSet/) problem. In the constructor, `weights` are the weights of vertices.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.
"""
struct MaximalIndependentSet{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    code::CT
    weights::WT
end

function MaximalIndependentSet(g::SimpleGraph; weights=UnWeighted(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa UnWeighted || length(weights) == nv(g)
    rawcode = EinCode(([[Graphs.neighbors(g, v)..., v] for v in Graphs.vertices(g)]...,), collect(Int, openvertices))
    MaximalIndependentSet(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), weights)
end

flavors(::Type{<:MaximalIndependentSet}) = [0, 1]
symbols(gp::MaximalIndependentSet) = [i for i in 1:length(getixsv(gp.code))]
get_weights(gp::MaximalIndependentSet, label) = [0, gp.weights[findfirst(==(label), symbols(gp))]]

function generate_tensors(fx, mi::MaximalIndependentSet)
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

