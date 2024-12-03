"""
$TYPEDEF

The [maximal independent set](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/MaximalIS/) problem. In the constructor, `weights` are the weights of vertices.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the vertices of the `graph`.
"""
energy_terms(gp::MaximalIS) = [[Graphs.neighbors(gp.graph, v)..., v] for v in Graphs.vertices(gp.graph)]
energy_tensors(x::T, c::MaximalIS) where T = [maximal_independent_set_tensor(_pow.(Ref(x), get_weights(c, i))..., degree(c.graph, i)+1) for i=1:nv(c.graph)]
extra_terms(::MaximalIS) = Vector{Int}[]
extra_tensors(::Type{T}, ::MaximalIS) where T = Array{T}[]
labels(gp::MaximalIS) = [1:nv(gp.graph)...]

# weights interface
get_weights(c::MaximalIS) = c.weights
get_weights(gp::MaximalIS, i::Int) = [0, gp.weights[i]]

function maximal_independent_set_tensor(a::T, b::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = a
    end
    t[1<<(d-1)+1] = b
    return t
end


"""
    is_maximal_independent_set(g::SimpleGraph, config)

Return true if `config` (a vector of boolean numbers as the mask of vertices) is a maximal independent set of graph `g`.
"""
is_maximal_independent_set(g::SimpleGraph, config) = !any(e->config[e.src] == 1 && config[e.dst] == 1, edges(g)) && all(w->config[w] == 1 || any(v->!iszero(config[v]), neighbors(g, w)), Graphs.vertices(g))