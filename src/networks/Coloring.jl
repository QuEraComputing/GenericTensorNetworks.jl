"""
    Coloring{K,CT<:AbstractEinsum} <: GraphProblem
    Coloring{K}(graph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

The [Vertex Coloring](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/Coloring.html) problem.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct Coloring{K,CT<:AbstractEinsum} <: GraphProblem
    code::CT
    nv::Int
end
Coloring{K}(code::ET, nv::Int) where {K,ET<:AbstractEinsum} = Coloring{K,ET}(code, nv)
# same network layout as independent set.
function Coloring{K}(g::SimpleGraph; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where K
    rawcode = EinCode(([[i] for i in Graphs.vertices(g)]..., # labels for vertex tensors
                       [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]...), collect(Int, openvertices))  # labels for edge tensors
    code = _optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier)
    Coloring{K}(code, nv(g))
end

flavors(::Type{<:Coloring{K}}) where K = collect(0:K-1)
get_weights(::Coloring{K}, i::Int) where K = ones(Int, K)
terms(gp::Coloring) = getixsv(gp.code)[1:gp.nv]
labels(gp::Coloring) = [1:gp.nv...]

function generate_tensors(x::T, c::Coloring{K}) where {K,T}
    ixs = getixsv(c.code)
    return add_labels!(map(1:length(ixs)) do i
        i <= c.nv ? coloringv(x, K) .^ get_weights(c, i) : coloringb(T, K)
    end, ixs, labels(c))
end

# coloring bond tensor
function coloringb(::Type{T}, k::Int) where T
    x = ones(T, k, k)
    for i=1:k
        x[i,i] = zero(T)
    end
    return x
end
# coloring vertex tensor
coloringv(x::T, k::Int) where T = fill(x, k)

# utilities
"""
    is_good_vertex_coloring(graph::SimpleGraph, config)

Returns true if the coloring specified by config is a valid one, i.e. does not violate the contraints of vertices of an edges having different colors.
"""
function is_good_vertex_coloring(graph::SimpleGraph, config)
    for e in edges(graph)
        config[e.src] == config[e.dst] && return false
    end
    return true
end