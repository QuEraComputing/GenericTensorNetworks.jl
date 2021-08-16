export mis_config, ConfigEnumerator, ConfigTropical

function symbols(::EinCode{ixs}) where ixs
    res = []
    for ix in ixs
        for l in ix
            if l ∉ res
                push!(res, l)
            end
        end
    end
    return res
end

function mis_config(code; all=false, bounding=true, usecuda=false)
    if all && usecuda
        throw(ArgumentError("ConfigEnumerator can not be computed on GPU!"))
    end
    flatten_code = flatten(code)
    syms = unique(Iterators.flatten(filter(x->length(x)==1,OMEinsum.getixs(flatten_code))))
    vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
    N = length(vertex_index)
    C = TropicalNumbers._nints(N)
    xs = map(getixs(flatten_code)) do ix
        T = all ? CountingTropical{Float64, ConfigEnumerator{N,C}} : ConfigTropical{Float64, N, C}
        if length(ix) == 2
            return misb(T)
        else
            s = TropicalNumbers.onehot(StaticBitVector{N,C}, vertex_index[ix[1]])
            if all
                misv(T, T(1.0, ConfigEnumerator([s])))
            else
                misv(T, T(1.0, s))
            end
        end
    end
    if bounding
        ymask = trues(fill(2, length(getiy(flatten_code)))...)
        xst = map(getixs(flatten_code)) do ix
            length(ix) == 1 ? misv(TropicalF64,Tropical(1.0)) : misb(TropicalF64)
        end
        if usecuda
            ymask = CuArray(ymask)
            xst = CuArray.(xst)
        end
        if all
            return bounding_contract(code, xst, ymask, xs)
        else
            @assert ndims(ymask) == 0
            t, res = mis_config_ad(code, xst, ymask)
            return fill(ConfigTropical(asscalar(t).n, StaticBitVector(map(l->res[l], 1:N))))
        end
    else
        if usecuda
            xs = CuArray.(xs)
        end
	    return dynamic_einsum(code, xs)
    end
end

export mis_max2_config
function mis_max2_config(code)
    flatten_code = flatten(code)
    syms = unique(Iterators.flatten(filter(x->length(x)==1,OMEinsum.getixs(flatten_code))))
    vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
    N = length(vertex_index)
    C = TropicalNumbers._nints(N)
    xs = map(getixs(flatten_code)) do ix
        T = Max2Poly{ConfigEnumerator{N,C}}
        if length(ix) == 2
            return misb(T)
        else
            s = TropicalNumbers.onehot(StaticBitVector{N,C}, vertex_index[ix[1]])
            misv(T, Max2Poly(zero(ConfigEnumerator{N,C}), ConfigEnumerator([s]), 1.0))
        end
    end
    return dynamic_einsum(code, xs)
end

export all_config
function all_config(code)
    flatten_code = flatten(code)
    syms = unique(Iterators.flatten(filter(x->length(x)==1,OMEinsum.getixs(flatten_code))))
    vertex_index = Dict([s=>i for (i, s) in enumerate(syms)])
    N = length(vertex_index)
    C = TropicalNumbers._nints(N)
    xs = map(getixs(flatten_code)) do ix
        T = Polynomial{ConfigEnumerator{N,C}, :x}
        if length(ix) == 2
            return misb(T)
        else
            s = TropicalNumbers.onehot(StaticBitVector{N,C}, vertex_index[ix[1]])
            misv(T, Polynomial([zero(ConfigEnumerator{N,C}), ConfigEnumerator([s])]))
        end
    end
    return dynamic_einsum(code, xs)
end