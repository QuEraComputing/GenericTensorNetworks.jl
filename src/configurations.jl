"""
    best_solutions(problem; all=false, usecuda=false, invert=false)
    
Find optimal solutions with bounding.

* When `all` is true, the program will use set for enumerate all possible solutions, otherwise, it will return one solution for each size.
* `usecuda` can not be true if you want to use set to enumerate all possible solutions.
* If `invert` is true, find the minimum.
"""
function best_solutions(gp::GraphProblem; all=false, usecuda=false, invert=false)
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    xst = generate_tensors(l->(vs = TropicalF64.(get_weights(gp, l)); invert ? pre_invert_exponent.(vs) : vs), gp)
    ymask = trues(fill(2, length(getiyv(gp.code)))...)
    if usecuda
        xst = CuArray.(xst)
        ymask = CuArray(ymask)
    end
    if all
        # we use `Float64` types because we want to support weighted graphs.
        xs = generate_tensors(fx_solutions(gp, CountingTropical{Float64,Float64}, all, invert), gp)
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
    solutions(problem, basetype; all, usecuda=false, invert=false)
    
General routine to find solutions without bounding,

* `basetype` can be a type with counting field,
    * `CountingTropical{Float64,Float64}` for finding optimal solutions,
    * `Polynomial{Float64, :x}` for enumerating all solutions,
    * `Max2Poly{Float64,Float64}` for optimal and suboptimal solutions.
* When `all` is true, the program will use set for enumerate all possible solutions, otherwise, it will return one solution for each size.
* `usecuda` can not be true if you want to use set to enumerate all possible solutions.
"""
function solutions(gp::GraphProblem, ::Type{BT}; all::Bool, usecuda::Bool=false, invert::Bool=false) where BT
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    ret = contractf(fx_solutions(gp, BT, all, invert), gp; usecuda=usecuda)
    return invert ? post_invert_exponent.(ret) : ret
end

"""
    best2_solutions(problem; all=true, usecuda=false, invert=false)

Finding optimal and suboptimal solutions.
"""
best2_solutions(gp::GraphProblem; all=true, usecuda=false, invert::Bool=false) = solutions(gp, Max2Poly{Float64,Float64}; all, usecuda, invert)

function bestk_solutions(gp::GraphProblem, k::Int; invert::Bool=false)
    xst = generate_tensors(l->TropicalF64.(get_weights(gp, l)), gp)
    ymask = trues(fill(2, length(getiyv(gp.code)))...)
    xs = generate_tensors(fx_solutions(gp, TruncatedPoly{k,Float64,Float64}, true, invert), gp)
    ret = bounding_contract(AllConfigs{k}(), gp.code, xst, ymask, xs)
    return invert ? post_invert_exponent.(ret) : ret
end

"""
    all_solutions(problem)

Finding all solutions grouped by size.
e.g. when the problem is [`MaximalIS`](@ref), it computes all maximal independent sets, or the maximal cliques of it complement.
"""
all_solutions(gp::GraphProblem) = solutions(gp, Polynomial{Float64,:x}, all=true, usecuda=false)

# return a mapping from label to onehot bitstrings (degree of freedoms).
function fx_solutions(gp::GraphProblem, ::Type{BT}, all::Bool, invert::Bool) where {BT}
    syms = symbols(gp)
    T = (all ? set_type : sampler_type)(BT, length(syms), nflavor(gp))
    vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
    return function (l)
        ret = _onehotv.(Ref(T), vertex_index[l], flavors(gp), get_weights(gp, l))
        invert ? pre_invert_exponent.(ret) : ret
    end
end
function _onehotv(::Type{Polynomial{BS,X}}, x, v, w) where {BS,X}
    Polynomial{BS,X}([zero(BS), onehotv(BS, x, v)])^w
end
function _onehotv(::Type{TruncatedPoly{K,BS,OS}}, x, v, w) where {K,BS,OS}
    TruncatedPoly{K,BS,OS}(ntuple(i->i<K ? zero(BS) : onehotv(BS, x, v), K),one(OS))^w
end
function _onehotv(::Type{CountingTropical{TV,BS}}, x, v, w) where {TV,BS}
    CountingTropical{TV,BS}(TV(w), onehotv(BS, x, v))
end
function _onehotv(::Type{BS}, x, v, w) where {BS<:ConfigEnumerator}
    onehotv(BS, x, v)
end