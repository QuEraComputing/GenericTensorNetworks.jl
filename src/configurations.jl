function config_type(::Type{T}, n, nflavor; all::Bool, tree_storage::Bool) where T
    if all
        if tree_storage
            return treeset_type(T, n, nflavor)
        else
            return set_type(T, n, nflavor)
        end
    else
        return sampler_type(T, n, nflavor)
    end
end

"""
    best_solutions(problem; all=false, usecuda=false, invert=false, tree_storage::Bool=false)
    
Find optimal solutions with bounding.

* When `all` is true, the program will use set for enumerate all possible solutions, otherwise, it will return one solution for each size.
* `usecuda` can not be true if you want to use set to enumerate all possible solutions.
* If `invert` is true, find the minimum.
* If `tree_storage` is true, use [`SumProductTree`](@ref) as the storage of solutions.
"""
function best_solutions(gp::GraphProblem; all=false, usecuda=false, invert=false, tree_storage::Bool=false, T=Float64)
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    xst = generate_tensors(_x(Tropical{T}; invert), gp)
    ymask = trues(fill(2, length(getiyv(gp.code)))...)
    if usecuda
        xst = togpu.(xst)
        ymask = togpu(ymask)
    end
    if all
        # we use `Float64` as default because we want to support weighted graphs.
        T = config_type(CountingTropical{T,T}, length(labels(gp)), nflavor(gp); all, tree_storage)
        xs = generate_tensors(_x(T; invert), gp)
        ret = bounding_contract(AllConfigs{1}(), gp.code, xst, ymask, xs)
        return invert ? asarray(post_invert_exponent.(ret), ret) : ret
    else
        @assert ndims(ymask) == 0
        t, res = solution_ad(gp.code, xst, ymask)
        ret = fill(CountingTropical(asscalar(t).n, ConfigSampler(StaticBitVector(map(l->res[l], 1:length(res))))))
        return invert ? asarray(post_invert_exponent.(ret), ret) : ret
    end
end

"""
    solutions(problem, basetype; all, usecuda=false, invert=false, tree_storage::Bool=false)
    
General routine to find solutions without bounding,

* `basetype` can be a type with counting field,
    * `CountingTropical{Float64,Float64}` for finding optimal solutions,
    * `Polynomial{Float64, :x}` for enumerating all solutions,
    * `Max2Poly{Float64,Float64}` for optimal and suboptimal solutions.
* When `all` is true, the program will use set for enumerate all possible solutions, otherwise, it will return one solution for each size.
* `usecuda` can not be true if you want to use set to enumerate all possible solutions.
* If `tree_storage` is true, use [`SumProductTree`](@ref) as the storage of solutions.
"""
function solutions(gp::GraphProblem, ::Type{BT}; all::Bool, usecuda::Bool=false, invert::Bool=false, tree_storage::Bool=false) where BT
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    T = config_type(BT, length(labels(gp)), nflavor(gp); all, tree_storage)
    ret = contractx(gp, _x(T; invert); usecuda=usecuda)
    return invert ? asarray(post_invert_exponent.(ret), ret) : ret
end

"""
    best2_solutions(problem; all=true, usecuda=false, invert=false, tree_storage::Bool=false)

Finding optimal and suboptimal solutions.
"""
best2_solutions(gp::GraphProblem; all=true, usecuda=false, invert::Bool=false, T=Float64) = solutions(gp, Max2Poly{T,T}; all, usecuda, invert)

function bestk_solutions(gp::GraphProblem, k::Int; invert::Bool=false, tree_storage::Bool=false, T=Float64)
    xst = generate_tensors(_x(Tropical{T}; invert), gp)
    ymask = trues(fill(2, length(getiyv(gp.code)))...)
    T = config_type(TruncatedPoly{k,T,T}, length(labels(gp)), nflavor(gp); all=true, tree_storage)
    xs = generate_tensors(_x(T; invert), gp)
    ret = bounding_contract(AllConfigs{k}(), gp.code, xst, ymask, xs)
    return invert ? asarray(post_invert_exponent.(ret), ret) : ret
end

"""
    all_solutions(problem)

Finding all solutions grouped by size.
e.g. when the problem is [`MaximalIS`](@ref), it computes all maximal independent sets, or the maximal cliques of it complement.
"""
all_solutions(gp::GraphProblem; T=Float64) = solutions(gp, Polynomial{T,:x}, all=true, usecuda=false, tree_storage=false)

# NOTE: do we have more efficient way to compute it?
# NOTE: doing pair-wise Hamming distance might be biased?
"""
    hamming_distribution(S, T)

Compute the distribution of pair-wise Hamming distances, which is defined as:
```math
c(k) := \\sum_{\\sigma\\in S, \\tau\\in T} \\delta({\\rm dist}(\\sigma, \\tau), k)
```
where ``\\delta`` is a function that returns 1 if two arguments are equivalent, 0 otherwise,
``{\\rm dist}`` is the Hamming distance function.

Returns the counting as a vector.
"""
function hamming_distribution(t1::ConfigEnumerator, t2::ConfigEnumerator)
    return hamming_distribution(t1.data, t2.data)
end
function hamming_distribution(s1::AbstractVector{StaticElementVector{N,S,C}}, s2::AbstractVector{StaticElementVector{N,S,C}}) where {N,S,C}
    return hamming_distribution!(zeros(Int, N+1), s1, s2)
end
function hamming_distribution!(out::AbstractVector, s1::AbstractVector{StaticElementVector{N,S,C}}, s2::AbstractVector{StaticElementVector{N,S,C}}) where {N,S,C}
    @assert length(out) == N+1
    @inbounds for a in s1, b in s2
        out[count_ones(a ⊻ b)+1] += 1
    end
    return out
end