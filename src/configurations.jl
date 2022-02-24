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
* If `tree_storage` is true, use [`TreeConfigEnumerator`](@ref) as the storage of solutions.
"""
function best_solutions(gp::GraphProblem; all=false, usecuda=false, invert=false, tree_storage::Bool=false)
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    xst = generate_tensors(_x(TropicalF64; invert), gp)
    ymask = trues(fill(2, length(getiyv(gp.code)))...)
    if usecuda
        xst = CuArray.(xst)
        ymask = CuArray(ymask)
    end
    if all
        # we use `Float64` types because we want to support weighted graphs.
        T = config_type(CountingTropical{Float64,Float64}, length(labels(gp)), nflavor(gp); all, tree_storage)
        xs = generate_tensors(_x(T; invert), gp)
        ret = bounding_contract(AllConfigs{1}(), gp.code, xst, ymask, xs)
        return invert ? post_invert_exponent.(ret) : ret
    else
        @assert ndims(ymask) == 0
        t, res = solution_ad(gp.code, xst, ymask)
        ret = fill(CountingTropical(asscalar(t).n, ConfigSampler(StaticBitVector(map(l->res[l], 1:length(res))))))
        return invert ? post_invert_exponent.(ret) : ret
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
* If `tree_storage` is true, use [`TreeConfigEnumerator`](@ref) as the storage of solutions.
"""
function solutions(gp::GraphProblem, ::Type{BT}; all::Bool, usecuda::Bool=false, invert::Bool=false, tree_storage::Bool=false) where BT
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    T = config_type(BT, length(labels(gp)), nflavor(gp); all, tree_storage)
    ret = contractx(gp, _x(T; invert); usecuda=usecuda)
    return invert ? post_invert_exponent.(ret) : ret
end

"""
    best2_solutions(problem; all=true, usecuda=false, invert=false, tree_storage::Bool=false)

Finding optimal and suboptimal solutions.
"""
best2_solutions(gp::GraphProblem; all=true, usecuda=false, invert::Bool=false) = solutions(gp, Max2Poly{Float64,Float64}; all, usecuda, invert)

function bestk_solutions(gp::GraphProblem, k::Int; invert::Bool=false, tree_storage::Bool=false)
    xst = generate_tensors(_x(TropicalF64; invert), gp)
    ymask = trues(fill(2, length(getiyv(gp.code)))...)
    T = config_type(TruncatedPoly{k,Float64,Float64}, length(labels(gp)), nflavor(gp); all=true, tree_storage)
    xs = generate_tensors(_x(T; invert), gp)
    ret = bounding_contract(AllConfigs{k}(), gp.code, xst, ymask, xs)
    return invert ? post_invert_exponent.(ret) : ret
end

"""
    all_solutions(problem)

Finding all solutions grouped by size.
e.g. when the problem is [`MaximalIS`](@ref), it computes all maximal independent sets, or the maximal cliques of it complement.
"""
all_solutions(gp::GraphProblem) = solutions(gp, Polynomial{Float64,:x}, all=true, usecuda=false, tree_storage=false)

function _onehotv(::Type{Polynomial{BS,X}}, x, v) where {BS,X}
    Polynomial{BS,X}([onehotv(BS, x, v)])
end
function _onehotv(::Type{TruncatedPoly{K,BS,OS}}, x, v) where {K,BS,OS}
    TruncatedPoly{K,BS,OS}(ntuple(i->i != K ? zero(BS) : onehotv(BS, x, v), K),zero(OS))
end
function _onehotv(::Type{CountingTropical{TV,BS}}, x, v) where {TV,BS}
    CountingTropical{TV,BS}(zero(TV), onehotv(BS, x, v))
end
function _onehotv(::Type{BS}, x, v) where {BS<:AbstractSetNumber}
    onehotv(BS, x, v)
end