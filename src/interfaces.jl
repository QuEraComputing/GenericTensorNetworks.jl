export solve

"""
    solve(problem, task; usecuda=false)

* `problem` is the graph problem with tensor network information,
* `task` is string specifying the task. Using the maximum independent set problem as an example, it can be one of
    * "size max", the maximum independent set size,
    * "counting sum", total number of independent sets,
    * "counting max", the dengeneracy of maximum independent sets (MIS),
    * "counting max2", the dengeneracy of MIS and MIS-1,
    * "counting all", independence polynomial, the polynomial number approach,
    * "counting all (fft)", independence polynomial, the fourier transformation approach,
    * "counting all (finitefield)", independence polynomial, the finite field approach,
    * "config max", one of the maximum independent set,
    * "config max (bounded)", one of the maximum independent set, the bounded version,
    * "configs max", all MIS configurations,
    * "configs max2", all MIS configurations and MIS-1 configurations,
    * "configs all", all IS configurations,
    * "configs max (bounded)", all MIS configurations, the bounded approach (much faster),
"""
function solve(gp::GraphProblem, task; usecuda=false, kwargs...)
    if task == "size max"
        return contractx(gp, Tropical(1.0); usecuda=usecuda)
    elseif task == "counting sum"
        return contractx(gp, 1.0; usecuda=usecuda)
    elseif task == "counting max"
        return contractx(gp, CountingTropical(1.0); usecuda=usecuda)
    elseif task == "counting max2"
        return contractx(gp, Max2Poly(0.0, 1.0, 1.0); usecuda=usecuda)
    elseif task == "counting all"
        return graph_polynomial(gp, Val(:polynomial); usecuda=usecuda)
    elseif task == "config max"
        return solutions(gp, CountingTropical{Float64,Float64}; all=false, usecuda=usecuda)
    elseif task == "configs max"
        return solutions(gp, CountingTropical{Float64,Float64}; all=true, usecuda=usecuda)
    elseif task == "configs max2"
        return solutions(gp, Max2Poly{Float64,Float64}; all=true, usecuda=usecuda)
    elseif task == "configs all"
        return solutions(gp, Polynomial{Float64,:x}; all=true, usecuda=usecuda)
    # extra methods
    elseif task == "counting all (fft)"
        return graph_polynomial(gp, Val(:fft); usecuda=usecuda, kwargs...)
    elseif task == "counting all (finitefield)"
        return graph_polynomial(gp, Val(:finitefield); usecuda=usecuda, kwargs...)
    elseif task == "config max (bounded)"
        return best_solutions(gp; all=false, usecuda=usecuda)
    elseif task == "configs max (bounded)"
        return best_solutions(gp; all=true, usecuda=usecuda)
    else
        error("unknown task $task.")
    end
end

export save_configs, load_configs
using DelimitedFiles
function save_configs(filename, data::ConfigEnumerator{N,S,C}; format::Symbol=:binary) where {N,S,C}
    if format == :binary
        write(filename, raw_matrix(data))
    elseif format == :text
        writedlm(filename, plain_matrix(data))
    else
        error("format must be `:binary` or `:text`, got `:$format`")
    end
end
function load_configs(filename; len=nothing, format::Symbol=:binary, nflavors=2)
    if format == :binary
        len === nothing && error("you need to specify `len` for reading configurations from binary files.")
        S = ceil(Int, log2(nflavors))
        C = _nints(len, S)
        return _from_raw_matrix(StaticElementVector{len,S,C}, reshape(reinterpret(UInt64, read(filename)),C,:))
    elseif format == :text
        return from_plain_matrix(readdlm(filename); nflavors=nflavors)
    else
        error("format must be `:binary` or `:text`, got `:$format`")
    end
end

function raw_matrix(x::ConfigEnumerator{N,S,C}) where {N,S,C}
    m = zeros(UInt64, C, length(x))
    @inbounds for i=1:length(x), j=1:C
        m[j,i] = x.data[i].data[j]
    end
    return m
end
function plain_matrix(x::ConfigEnumerator{N,S,C}) where {N,S,C}
    m = zeros(UInt8, N, length(x))
    @inbounds for i=1:length(x), j=1:N
        m[j,i] = x.data[i][j]
    end
    return m
end

function from_raw_matrix(m; len, nflavors=2)
    S = ceil(Int,log2(nflavors))
    C = size(m, 1)
    T = StaticElementVector{len,S,C}
    @assert len*S <= C*64
    _from_raw_matrix(T, m)
end
function _from_raw_matrix(::Type{StaticElementVector{N,S,C}}, m::AbstractMatrix) where {N,S,C}
    data = zeros(StaticElementVector{N,S,C}, size(m, 2))
    @inbounds for i=1:size(m, 2)
        data[i] = StaticElementVector{N,S,C}(NTuple{C,UInt64}(view(m,:,i)))
    end
    return ConfigEnumerator(data)
end
function from_plain_matrix(m::Matrix; nflavors=2)
    S = ceil(Int,log2(nflavors))
    N = size(m, 1)
    C = _nints(N, S)
    T = StaticElementVector{N,S,C}
    _from_plain_matrix(T, m)
end
function _from_plain_matrix(::Type{StaticElementVector{N,S,C}}, m::AbstractMatrix) where {N,S,C}
    data = zeros(StaticElementVector{N,S,C}, size(m, 2))
    @inbounds for i=1:size(m, 2)
        data[i] = convert(StaticElementVector{N,S,C}, view(m, :, i))
    end
    return ConfigEnumerator(data)
end

# convert to Matrix
Base.Matrix(ce::ConfigEnumerator) = plain_matrix(ce)
Base.Vector(ce::StaticElementVector) = collect(ce)