export optimalsolutions, solutions

"""
    optimalsolutions(problem; all=false, usecuda=false)
    
Find optimal solutions with bounding.

* When `all` is true, the program will use set for enumerate all possible solutions, otherwise, it will return one solution for each size.
* `usecuda` can not be true if you want to use set to enumerate all possible solutions.
"""
function optimalsolutions(gp::GraphProblem; all=false, usecuda=false)
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    syms = symbols(gp)
    T = (all ? set_type : sampler_type)(CountingTropical{Int64}, length(syms), bondsize(gp))
    vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
    xst = generate_tensors(l->TropicalF64(1.0), gp)
    ymask = trues(fill(2, length(OMEinsum.getiy(flatten(gp.code))))...)
    if usecuda
        xst = CuArray.(xst)
        ymask = CuArray(ymask)
    end
    if all
        xs = generate_tensors(l->onehotv(T, vertex_index[l], 1), gp)
        return bounding_contract(gp.code, xst, ymask, xs)
    else
        @assert ndims(ymask) == 0
        t, res = solution_ad(gp.code, xst, ymask)
        N = length(vertex_index)
        return fill(CountingTropical(asscalar(t).n, ConfigSampler(StaticBitVector(map(l->res[l], 1:N)))))
    end
end

"""
    solutions(problem, basetype; all=false, usecuda=false)
    
General routine to find solutions without bounding,

* `basetype` can be a type with counting field,
    * `CountingTropical{Float64,Float64}` for finding optimal solutions,
    * `Polynomial{Float64, 1.0}` for enumerating all solutions,
    * `Max2Poly{Float64, 1.0}` for optimal and suboptimal solutions.
* When `all` is true, the program will use set for enumerate all possible solutions, otherwise, it will return one solution for each size.
* `usecuda` can not be true if you want to use set to enumerate all possible solutions.
"""
function solutions(gp::GraphProblem, ::Type{BT}; all=false, usecuda=false) where BT
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    return contractf(fx_solutions(gp, BT, all), gp; usecuda=usecuda)
end

# return a mapping from label to variable `x`
for GP in [:Independence, :Matching, :MaximalIndependence, :MaxCut]
    @eval function fx_solutions(gp::$GP, ::Type{BT}, all::Bool) where BT
        syms = symbols(gp)
        T = (all ? set_type : sampler_type)(BT, length(syms), bondsize(gp))
        vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
        return l->onehotv(T, vertex_index[l], 1)
    end
end
function fx_solutions(gp::Coloring{K}, ::Type{BT}, all::Bool) where {K,BT}
    syms = symbols(gp)
    T = (all ? set_type : sampler_type)(BT, length(syms), bondsize(gp))
    vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
    return function (l)
        map(1:K) do k
            onehotv(T, vertex_index[l], k)
        end
    end
end

for GP in [:Independence, :Matching, :MaximalIndependence, :Coloring]
    @eval symbols(gp::$GP) = labels(gp.code)
end
symbols(gp::MaxCut) = collect(OMEinsum.getixs(OMEinsum.flatten(gp.code)))
# TODO: coloring