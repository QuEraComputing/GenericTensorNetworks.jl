export getconfigs_bounded, getconfigs_direct

function getconfigs_bounded(gp::GraphProblem; all=false, usecuda=false)
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    T = (all ? bitstringset_type : bitstringsampler_type)(CountingTropical{Int64}, length(labels(gp.code)))
    syms = labels(gp.code)
    vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
    xst = generate_tensors(l->TropicalF64(1.0), gp)
    ymask = trues(fill(2, length(OMEinsum.getiy(flatten(gp.code))))...)
    if usecuda
        xst = CuArray.(xst)
        ymask = CuArray(ymask)
    end
    if all
        xs = generate_tensors(l->onehotv(T, vertex_index[l]), gp)
        return bounding_contract(gp.code, xst, ymask, xs)
    else
        @assert ndims(ymask) == 0
        t, res = mis_config_ad(gp.code, xst, ymask)
        N = length(vertex_index)
        return fill(CountingTropical(asscalar(t).n, ConfigSampler(StaticBitVector(map(l->res[l], 1:N)))))
    end
end

function getconfigs_direct(gp::GraphProblem; all=false, usecuda=false)
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    T = (all ? bitstringset_type : bitstringsampler_type)(CountingTropical{Int64}, length(labels(gp.code)))
    syms = labels(gp.code)
    vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
    xs = generate_tensors(l->onehotv(T, vertex_index[l]), gp)
    if usecuda
        xs = CuArray.(xs)
    end
    dynamic_einsum(gp.code, xs)
end