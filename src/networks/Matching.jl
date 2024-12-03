"""
$TYPEDEF

The [Vertex matching](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Matching/) problem.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the edges of the `graph`.
"""
energy_terms(gp::Matching) = [[minmax(src(s), dst(s))] for s in edges(gp.graph)] # edge tensors
energy_tensors(x::T, c::Matching) where T = [_pow.(Ref(x), get_weights(c, i)) for i=1:ne(c.graph)]
extra_terms(gp::Matching) = [[minmax(i, j) for j in neighbors(gp.graph, i)] for i in Graphs.vertices(gp.graph)]
extra_tensors(::Type{T}, gp::Matching) where T = [match_tensor(T, degree(gp.graph, i)) for i=1:nv(gp.graph)]
labels(gp::Matching) = getindex.(energy_terms(gp))

# weights interface
get_weights(c::Matching) = c.weights
get_weights(m::Matching, i::Int) = [0, m.weights[i]]

function match_tensor(::Type{T}, n::Int) where T
    t = zeros(T, fill(2, n)...)
    for ci in CartesianIndices(t)
        if sum(ci.I .- 1) <= 1
            t[ci] = one(T)
        end
    end
    return t
end