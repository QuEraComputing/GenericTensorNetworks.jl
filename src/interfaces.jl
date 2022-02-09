abstract type AbstractProperty end

"""
    SizeMax <: AbstractProperty
    SizeMax()

The maximum independent set size.

* The corresponding tensor element type is [`Tropical`](@ref).
* It is compatible with weighted graph problems.
* BLAS (on CPU) and GPU are supported,
"""
struct SizeMax <: AbstractProperty end

"""
    CountingAll <: AbstractProperty
    CountingAll()

Counting the total number of sets. e.g. for [`IndependentSet`](@ref) problem, it counts the independent sets.

* The corresponding tensor element type is `Base.Real`.
* The weights on graph does not have effect.
* BLAS (GPU and CPU) and GPU are supported,
"""
struct CountingAll <: AbstractProperty end

"""
    CountingMax{K} <: AbstractProperty
    CountingMax(K=1)

Counting the number of sets with `K` largest size. e.g. for [`IndependentSet`](@ref) problem,
it counts independent sets of size ``\\alpha(G), \\alpha(G)-1, \\ldots, \\alpha(G)-K+1``.

* The corresponding tensor element type is [`CountingTropical`](@ref) for `K == 1`, and [`TruncatedPoly`](@ref)`{K}` for `K > 1`.
* Weighted graph problems is only supported for `K == 1`.
* GPU is supported,
"""
struct CountingMax{K} <: AbstractProperty end
CountingMax(K::Int=1) = CountingMax{K}()
max_k(::CountingMax{K}) where K = K

"""
    GraphPolynomial{METHOD} <: AbstractProperty
    GraphPolynomial(; method=:finitefield, kwargs...)

Compute the graph polynomial, e.g. for [`IndependentSet`](@ref) problem, it is the independence polynomial.
The `METHOD` type parameter can be one of the following symbols

* `:finitefield`, it uses finite field algebra to fit the polynomial.
    * The corresponding tensor element type is [`Mods.Mod`](@ref),
    * It does not have round-off error,
    * GPU is supported,
    * It accepts keyword arguments `maxorder` (optional, e.g. the MIS size in the [`IndependentSet`](@ref) problem).
* `:polynomial`, the program uses polynomial numbers to solve the polynomial directly.
    * The corresponding tensor element type is [`Polynomials.Polynomial`](@ref).
    * It might have small round-off error depending on the data type for storing the counting.
    * It has memory overhead that linear to the graph size.
* `:fft`, 
    * The corresponding tensor element type is `Base.Complex`.
    * It has (controllable) round-off error.
    * BLAS and GPU are supported.
    * It accepts keyword arguments `maxorder` (optional) and `r`,
        if `r > 1`, one has better precision for coefficients of large order, if `r < 1`,
        one has better precision for coefficients of small order.

Graph polynomials are not defined for weighted graph problems.
"""
struct GraphPolynomial{METHOD} <: AbstractProperty
    kwargs
end
GraphPolynomial(; method::Symbol = :finitefield, kwargs...) = GraphPolynomial{method}(kwargs)
graph_polynomial_method(::GraphPolynomial{METHOD}) where METHOD = METHOD

"""
    SingleConfigMax{BOUNDED} <: AbstractProperty
    SingleConfigMax(; bounded=false)

Finding single best solution, e.g. for [`IndependentSet`](@ref) problem, it is one of the maximum independent sets.

* The corresponding data type is [`CountingTropical{Float64,<:ConfigSampler}`](@ref) if `BOUNDED` is `true`, [`Tropical`](@ref) otherwise.
* Weighted graph problems is supported.
* GPU is supported,
"""
struct SingleConfigMax{BOUNDED} <:AbstractProperty end
SingleConfigMax(; bounded::Bool=false) = SingleConfigMax{bounded}()

"""
    ConfigsAll <:AbstractProperty
    ConfigsAll()

Find all valid configurations, e.g. for [`IndependentSet`](@ref) problem, it is finding all independent sets.

* The corresponding data type is [`ConfigEnumerator`](@ref).
* Weights do not take effect.
"""
struct ConfigsAll <:AbstractProperty end

"""
    ConfigsMax{K, BOUNDED} <:AbstractProperty
    ConfigsMax(K=1; bounded=true)

Find configurations with largest sizes, e.g. for [`IndependentSet`](@ref) problem,
it is finding all independent sets of sizes ``\\alpha(G), \\alpha(G)-1, \\ldots, \\alpha(G)-K+1``.

* The corresponding data type is [`CountingTropical`](@ref)`{Float64,<:ConfigEnumerator}` for `K == 1` and [`TruncatedPoly`](@ref)`{K,<:ConfigEnumerator}` for `K > 1`.
* Weighted graph problems is only supported for `K == 1`.
"""
struct ConfigsMax{K, BOUNDED} <:AbstractProperty end
ConfigsMax(K::Int=1; bounded::Bool=true) = ConfigsMax{K,bounded}()
max_k(::ConfigsMax{K}) where K = K

"""
    solve(problem, property; usecuda=false, T=Float64)

Solving a certain property of a graph problem.

Positional Arguments
---------------------------
* `problem` is the graph problem with tensor network information,
* `property` is string specifying the task. Using the maximum independent set problem as an example, it can be one of

    * [`SizeMax`](@ref) for finding maximum configuration size,

    * [`CountingMax`](@ref) for counting configurations with top `K` sizes,
    * [`CountingAll`](@ref) for counting all configurations,
    * [`GraphPolynomial`](@ref) for evaluating the graph polynomial,

    * [`SingleConfigMax`](@ref) for finding one maximum configuration,
    * [`ConfigsMax`](@ref) for enumerating configurations with top `K` sizes,
    * [`ConfigsAll`](@ref) for enumerating all configurations,


Keyword arguments
-------------------------------------
* `usecuda` is a switch to use CUDA (if possible), user need to call statement `using CUDA` before turning on this switch.
* `T` is the "base" element type, sometimes can be used to reduce the memory cost.
"""
function solve(gp::GraphProblem, property::AbstractProperty; T=Float64, usecuda=false)
    if property isa SizeMax
        syms = symbols(gp)
        return contractf(x->Tropical{T}.(get_weights(gp, x)), gp; usecuda=usecuda)
    elseif property isa CountingAll
        return contractx(gp, one(T); usecuda=usecuda)
    elseif property isa CountingMax{1}
        syms = symbols(gp)
        return contractf(x->CountingTropical{T,T}.(get_weights(gp, x)), gp; usecuda=usecuda)
    elseif property isa CountingMax
        return contractx(gp, TruncatedPoly(ntuple(i->i == max_k(property) ? one(T) : zero(T), max_k(property)), one(T)); usecuda=usecuda)
    elseif property isa GraphPolynomial
        return graph_polynomial(gp, Val(graph_polynomial_method(property)); usecuda=usecuda, property.kwargs...)
    elseif property isa SingleConfigMax{false}
        return solutions(gp, CountingTropical{T,T}; all=false, usecuda=usecuda)
    elseif property isa ConfigsMax{1,false}
        return solutions(gp, CountingTropical{T,T}; all=true, usecuda=usecuda)
    elseif property isa (ConfigsMax{K, false} where K)
        return solutions(gp, TruncatedPoly{max_k(property),T,T}; all=true, usecuda=usecuda)
    elseif property isa ConfigsAll
        return solutions(gp, Real; all=true, usecuda=usecuda)
    elseif property isa SingleConfigMax{true}
        return best_solutions(gp; all=false, usecuda=usecuda)
    elseif property isa ConfigsMax{1,true}
        return best_solutions(gp; all=true, usecuda=usecuda)
    elseif property isa (ConfigsMax{K,true} where K)
        return bestk_solutions(gp, max_k(property))
    else
        error("unknown property $property.")
    end
end

"""
    max_size(problem; usecuda=false)

Returns the maximum size of the graph problem. 
A shorthand of `solve(problem, SizeMax(); usecuda=false)`.
"""
function max_size end
"""
    max_size_count(problem; usecuda=false)

Returns the maximum size and the counting of the graph problem.
It is a shorthand of `solve(problem, CountingMax(); usecuda=false)`.
"""
function max_size_count end
for TP in [:MaximalIS, :IndependentSet, :Matching, :MaxCut, :PaintShop]
    @eval max_size(m::$TP; usecuda=false) = Int(sum(solve(m, SizeMax(); usecuda=usecuda)).n)  # floating point number is faster (BLAS)
    @eval max_size_count(m::$TP; usecuda=false) = (r = sum(solve(m, CountingMax(); usecuda=usecuda)); (Int(r.n), Int(r.c)))
end

using DelimitedFiles

"""
    save_configs(filename, data::ConfigEnumerator; format=:binary)

Save configurations `data` to file `filename`. The format is `:binary` or `:text`.
"""
function save_configs(filename, data::ConfigEnumerator{N,S,C}; format::Symbol=:binary) where {N,S,C}
    if format == :binary
        write(filename, raw_matrix(data))
    elseif format == :text
        writedlm(filename, plain_matrix(data))
    else
        error("format must be `:binary` or `:text`, got `:$format`")
    end
end

"""
    load_configs(filename; format=:binary, bitlength=nothing, nflavors=2)

Load configurations from file `filename`. The format is `:binary` or `:text`.
If the format is `:binary`, the bitstring length `bitlength` must be specified,
`nflavors` specifies the degree of freedom.
"""
function load_configs(filename; bitlength=nothing, format::Symbol=:binary, nflavors=2)
    if format == :binary
        bitlength === nothing && error("you need to specify `bitlength` for reading configurations from binary files.")
        S = ceil(Int, log2(nflavors))
        C = _nints(bitlength, S)
        return _from_raw_matrix(StaticElementVector{bitlength,S,C}, reshape(reinterpret(UInt64, read(filename)),C,:))
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

function from_raw_matrix(m; bitlength, nflavors=2)
    S = ceil(Int,log2(nflavors))
    C = size(m, 1)
    T = StaticElementVector{bitlength,S,C}
    @assert bitlength*S <= C*64
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