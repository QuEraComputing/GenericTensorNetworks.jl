"""
$TYPEDEF

The [independent set problem](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/IndependentSet/) in graph theory.

Positional arguments
-------------------------------
* `graph` is the problem graph.
* `weights` are associated with the vertices of the `graph`, default to `UnitWeight()`.

Examples
-------------------------------
```jldoctest; setup=:(using Random; Random.seed!(2))
julia> using GenericTensorNetworks, Graphs

julia> problem = independent_set_network(smallgraph(:petersen));

julia> solve(problem, ConfigsMax())
0-dimensional Array{CountingTropical{Float64, ConfigEnumerator{10, 1, 1}}, 0}:
(4.0, {0101010001, 1010000011, 0100100110, 0010111000, 1001001100})ₜ
```
"""
struct IndependentSet{WT} <: GraphProblem
    graph::SimpleGraph{Int}
    weights::WT
    function IndependentSet(graph::SimpleGraph{Int}, weights::WT=UnitWeight()) where WT
        @assert weights isa UnitWeight || length(weights) == nv(graph) "got unexpected weights for $(nv(graph))-vertex graph: $weights"
        new{WT}(graph, weights)
    end
end
function independent_set_network(g::SimpleGraph; weights=UnitWeight(), openvertices=(), fixedvertices=Dict{Int,Int}(), optimizer=GreedyMethod(), simplifier=nothing)
    cfg = IndependentSet(g, weights)
    gtn = GenericTensorNetwork(cfg; openvertices, fixedvertices)
    return OMEinsum.optimize_code(gtn; optimizer, simplifier)
end
flavors(::Type{<:IndependentSet}) = [0, 1]
energy_terms(gp::IndependentSet) = [[i] for i in 1:nv(gp.graph)]
energy_tensors(x::T, c::IndependentSet) where T = [misv(_pow.(Ref(x), get_weights(c, i))) for i=1:nv(c.graph)]
extra_terms(gp::IndependentSet) = [[minmax(e.src,e.dst)...] for e in Graphs.edges(gp.graph)]
extra_tensors(::Type{T}, gp::IndependentSet) where T = [misb(T) for i=1:ne(gp.graph)]
labels(gp::IndependentSet) = [1:nv(gp.graph)...]

# weights interface
get_weights(c::IndependentSet) = c.weights
get_weights(gp::IndependentSet, i::Int) = [0, gp.weights[i]]
chweights(c::IndependentSet, weights) = IndependentSet(c.graph, weights)

function misb(::Type{T}, n::Integer=2) where T
    res = zeros(T, fill(2, n)...)
    res[1] = one(T)
    for i=1:n
        res[1+1<<(i-1)] = one(T)
    end
    return res
end
misv(vals) = vals

"""
    mis_compactify!(tropicaltensor)

Compactify tropical tensor for maximum independent set problem. It will eliminate
some entries by setting them to zero, by the criteria that removing these entry
does not change the MIS size of its parent graph (reference to be added).
"""
function mis_compactify!(a::AbstractArray{T}) where T <: TropicalTypes
	for (ind_a, val_a) in enumerate(a)
		for (ind_b, val_b) in enumerate(a)
			bs_a = ind_a - 1
			bs_b = ind_b - 1
			@inbounds if bs_a != bs_b && val_a <= val_b && (bs_b & bs_a) == bs_b
				a[ind_a] = zero(T)
			end
		end
	end
	return a
end

"""
    is_independent_set(g::SimpleGraph, config)

Return true if `config` (a vector of boolean numbers as the mask of vertices) is an independent set of graph `g`.
"""
is_independent_set(g::SimpleGraph, config) = !any(e->config[e.src] == 1 && config[e.dst] == 1, edges(g))
