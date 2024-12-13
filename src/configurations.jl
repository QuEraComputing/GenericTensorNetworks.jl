function config_type(::Type{T}, n, num_flavors; all::Bool, tree_storage::Bool) where T
    if all
        if tree_storage
            return treeset_type(T, n, num_flavors)
        else
            return set_type(T, n, num_flavors)
        end
    else
        return sampler_type(T, n, num_flavors)
    end
end

"""
    largest_solutions(net::GenericTensorNetwork; all=false, usecuda=false, invert=false, tree_storage::Bool=false, T=Float64)
    
Find optimal solutions, with bounding. Please check [`solutions`](@ref) for argument descriptions.
"""
function largest_solutions(net::GenericTensorNetwork; all=false, usecuda=false, invert=false, tree_storage::Bool=false, T=Float64)
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    xst = generate_tensors(_x(Tropical{T}; invert), net)
    ymask = trues(fill(num_flavors(net), length(getiyv(net.code)))...)
    if usecuda
        xst = togpu.(xst)
        ymask = togpu(ymask)
    end
    if all
        # we use `Float64` as default because we want to support weighted graphs.
        T = config_type(CountingTropical{T,T}, length(variables(net)), num_flavors(net); all, tree_storage)
        xs = generate_tensors(_x(T; invert), net)
        ret = bounding_contract(AllConfigs{1}(), net.code, xst, ymask, xs)
        return invert ? asarray(post_invert_exponent.(ret), ret) : ret
    else
        @assert ndims(ymask) == 0
        t, res = solution_ad(net.code, xst, ymask)
        ret = fill(CountingTropical(asscalar(t).n, ConfigSampler(StaticBitVector(map(l->res[l], 1:length(res))))))
        return invert ? asarray(post_invert_exponent.(ret), ret) : ret
    end
end

"""
    solutions(net::GenericTensorNetwork, ::Type{BT}; all::Bool, usecuda::Bool=false, invert::Bool=false, tree_storage::Bool=false) where BT
    
Find all solutions, solutions with largest sizes or solutions with smallest sizes. Bounding is not supported.

### Arguments
- `net` is a [`GenericTensorNetwork`](@ref) instance.
- `BT` is the data types used for computing, which can be
    * `CountingTropical{Float64,Float64}` for finding optimal solutions,
    * `Polynomial{Float64, :x}` for enumerating all solutions,
    * `Max2Poly{Float64,Float64}` for optimal and suboptimal solutions.

### Keyword arguments
- `all` is an indicator whether to find all solutions or just one of them.
- `usecuda` is an indicator of using CUDA or not, which must be false if `all` is true.
- `invert` is an indicator of whether flip the size or not. If true, instead of finding the maximum, it find the minimum.
- `tree_storage` is an indicator of whether using more compact [`SumProductTree`](@ref) as the storage or not.
"""
function solutions(net::GenericTensorNetwork, ::Type{BT}; all::Bool, usecuda::Bool=false, invert::Bool=false, tree_storage::Bool=false) where BT
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    T = config_type(BT, length(variables(net)), num_flavors(net); all, tree_storage)
    ret = contractx(net, _x(T; invert); usecuda=usecuda)
    return invert ? asarray(post_invert_exponent.(ret), ret) : ret
end

function largestk_solutions(net::GenericTensorNetwork, k::Int; invert::Bool=false, tree_storage::Bool=false, T=Float64)
    xst = generate_tensors(_x(Tropical{T}; invert), net)
    ymask = trues(fill(2, length(getiyv(net.code)))...)
    T = config_type(TruncatedPoly{k,T,T}, length(variables(net)), num_flavors(net); all=true, tree_storage)
    xs = generate_tensors(_x(T; invert), net)
    ret = bounding_contract(AllConfigs{k}(), net.code, xst, ymask, xs)
    return invert ? asarray(post_invert_exponent.(ret), ret) : ret
end

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
        out[hamming_distance(a, b)+1] += 1
    end
    return out
end