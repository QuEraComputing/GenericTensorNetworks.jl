"""
    IndependentSet{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    IndependentSet(graph; weights=NoWeight(), openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

The [independent set problem](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/IndependentSet.html) in graph theory.
In the constructor, `weights` are the weights of vertices.
`openvertices` specifies labels for the output tensor.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct IndependentSet{CT<:AbstractEinsum,WT<:Union{NoWeight, Vector}} <: GraphProblem
    code::CT
    graph::SimpleGraph{Int}
    weights::WT
end

function IndependentSet(g::SimpleGraph; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    @assert weights isa NoWeight || length(weights) == nv(g)
    rawcode = EinCode([[[i] for i in Graphs.vertices(g)]..., # labels for vertex tensors
                       [[minmax(e.src,e.dst)...] for e in Graphs.edges(g)]...], collect(Int, openvertices))  # labels for edge tensors
    code = _optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier)
    IndependentSet(code, g, weights)
end

flavors(::Type{<:IndependentSet}) = [0, 1]
get_weights(gp::IndependentSet, i::Int) = [0, gp.weights[i]]
terms(gp::IndependentSet) = getixsv(gp.code)[1:nv(gp.graph)]
labels(gp::IndependentSet) = [1:nv(gp.graph)...]

# generate tensors
function generate_tensors(x::T, gp::IndependentSet) where T
    nv(gp.graph) == 0 && return []
    ixs = getixsv(gp.code)
    # we only add labels at vertex tensors
    return vcat(add_labels!([misv(Ref(x) .^ get_weights(gp, i)) for i=1:nv(gp.graph)], ixs[1:nv(gp.graph)], labels(gp)),
            [misb(T, length(ix)) for ix in ixs[nv(gp.graph)+1:end]] # if n!=2, it corresponds to set packing problem.
    )
end

function misb(::Type{T}, n::Integer=2) where T
    res = zeros(T, fill(2, n)...)
    res[1] = one(T)
    for i=1:n
        res[1+1<<(i-1)] = one(T)
    end
    return res
end
misv(vals) = vals

############### set packing #####################
"""
set_packing(sets; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)

Set packing is a generalization of independent set problem to hypergraphs.
Calling this function will return you an `IndependentSet` instance.
`sets` are a vector of vectors, each element being a vertex in the independent set problem.
`optimizer` and `simplifier` are for tensor network optimization, check `optimize_code` for details.

Example
-----------------------------------
```julia
julia> sets = [[1, 2, 5], [1, 3], [2, 4], [3, 6], [2, 3, 6]];  # each set is a vertex

julia> gp = set_packing(sets);

julia> res = best_solutions(gp; all=true)[]
(2, {10010, 00110, 01100})ₜ
```
"""
function set_packing(sets; weights=NoWeight(), openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    n = length(sets)
    code = EinCode(vcat([[i] for i=1:n], [[i,j] for i=1:n,j=1:n if j>i && !isempty(sets[i] ∩ sets[j])]), collect(Int,openvertices))
    # NOTE: we use a dummy graph here, which should be a hypergraph!
    IndependentSet(_optimize_code(code, uniformsize(code, 2), optimizer, simplifier), SimpleGraph(n), weights)
end

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
