abstract type AbstractProperty end

"""
    SizeMax <: AbstractProperty
    SizeMax()

The maximum set size. e.g. the largest size of the [`IndependentSet`](@ref)  problem is also know as the independence number.

* The corresponding tensor element type is max-plus tropical number [`Tropical`](@ref).
* It is compatible with weighted graph problems.
* BLAS (on CPU) and GPU are supported,
"""
struct SizeMax <: AbstractProperty end

"""
    SizeMin <: AbstractProperty
    SizeMin()

The maximum set size. e.g. the smallest size ofthe [`MaximalIS`](@ref) problem is also known as the independent domination number.

* The corresponding tensor element type inverted max-plus tropical number [`Tropical`](@ref), which is equivalent to the min-plus tropical number.
* It is compatible with weighted graph problems.
* BLAS (on CPU) and GPU are supported,
"""
struct SizeMin <: AbstractProperty end

"""
    CountingAll <: AbstractProperty
    CountingAll()

Counting the total number of sets. e.g. for the [`IndependentSet`](@ref) problem, it counts the independent sets.

* The corresponding tensor element type is `Base.Real`.
* The weights on graph does not have effect.
* BLAS (GPU and CPU) and GPU are supported,
"""
struct CountingAll <: AbstractProperty end

"""
    CountingMax{K} <: AbstractProperty
    CountingMax(K=1)

Counting the number of sets with largest-K size. e.g. for [`IndependentSet`](@ref) problem,
it counts independent sets of size ``\\alpha(G), \\alpha(G)-1, \\ldots, \\alpha(G)-K+1``.

* The corresponding tensor element type is [`CountingTropical`](@ref) for `K == 1`, and [`TruncatedPoly`](@ref)`{K}` for `K > 1`.
* Weighted graph problems is only supported for `K == 1`.
* GPU is supported,
"""
struct CountingMax{K} <: AbstractProperty end
CountingMax(K::Int=1) = CountingMax{K}()
max_k(::CountingMax{K}) where K = K

"""
    CountingMin{K} <: AbstractProperty
    CountingMin(K=1)

Counting the number of sets with smallest-K size.

* The corresponding tensor element type is inverted [`CountingTropical`](@ref) for `K == 1`, and [`TruncatedPoly`](@ref)`{K}` for `K > 1`.
* Weighted graph problems is only supported for `K == 1`.
* GPU is supported,
"""
struct CountingMin{K} <: AbstractProperty end
CountingMin(K::Int=1) = CountingMin{K}()
min_k(::CountingMin{K}) where K = K

"""
    GraphPolynomial{METHOD} <: AbstractProperty
    GraphPolynomial(; method=:finitefield, kwargs...)

Compute the graph polynomial, e.g. for [`IndependentSet`](@ref) problem, it is the independence polynomial.
The `METHOD` type parameter can be one of the following symbols

Method Argument
---------------------------
* `:finitefield`, uses finite field algebra to fit the polynomial.
    * The corresponding tensor element type is [`Mods.Mod`](@ref),
    * It does not have round-off error,
    * GPU is supported,
    * It accepts keyword arguments `maxorder` (optional, e.g. the MIS size in the [`IndependentSet`](@ref) problem).
* `:polynomial`, use polynomial numbers to solve the polynomial directly.
    * The corresponding tensor element type is [`Polynomials.Polynomial`](@ref).
    * It might have small round-off error depending on the data type for storing the counting.
    * It has memory overhead that linear to the graph size.
* `:fft`, use fast fourier transformation to fit the polynomial.
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

* The corresponding data type is [`CountingTropical{Float64,<:ConfigSampler}`](@ref) if `BOUNDED` is `false`, [`Tropical`](@ref) otherwise.
* Weighted graph problems is supported.
* GPU is supported,

Keyword Arguments
----------------------------
* `bounded`, if it is true, use bounding trick (or boolean gradients) to reduce the working memory to store intermediate configurations.
"""
struct SingleConfigMax{BOUNDED} <:AbstractProperty end
SingleConfigMax(; bounded::Bool=false) = SingleConfigMax{bounded}()

"""
    SingleConfigMin{BOUNDED} <: AbstractProperty
    SingleConfigMin(; bounded=false)

Finding single "worst" solution.

* The corresponding data type is inverted [`CountingTropical{Float64,<:ConfigSampler}`](@ref) if `BOUNDED` is `false`, inverted [`Tropical`](@ref) otherwise.
* Weighted graph problems is supported.
* GPU is supported,

Keyword Arguments
----------------------------
* `bounded`, if it is true, use bounding trick (or boolean gradients) to reduce the working memory to store intermediate configurations.
"""
struct SingleConfigMin{BOUNDED} <:AbstractProperty end
SingleConfigMin(; bounded::Bool=false) = SingleConfigMin{bounded}()

"""
    ConfigsAll{TREESTORAGE} <:AbstractProperty
    ConfigsAll(; tree_storage=false)

Find all valid configurations, e.g. for [`IndependentSet`](@ref) problem, it is finding all independent sets.

* The corresponding data type is [`ConfigEnumerator`](@ref).
* Weights do not take effect.

Keyword Arguments
----------------------------
* `tree_storage`, if it is true, it uses more memory efficient tree-structure to store the configurations.
"""
struct ConfigsAll{TREESTORAGE} <:AbstractProperty end
ConfigsAll(; tree_storage::Bool=false) = ConfigsAll{tree_storage}()
tree_storage(::ConfigsAll{TREESTORAGE}) where {TREESTORAGE} = TREESTORAGE

"""
    ConfigsMax{K, BOUNDED, TREESTORAGE} <:AbstractProperty
    ConfigsMax(K=1; bounded=true, tree_storage=true)

Find configurations with largest-K sizes, e.g. for [`IndependentSet`](@ref) problem,
it is finding all independent sets of sizes ``\\alpha(G), \\alpha(G)-1, \\ldots, \\alpha(G)-K+1``.

* The corresponding data type is [`CountingTropical`](@ref)`{Float64,<:ConfigEnumerator}` for `K == 1` and [`TruncatedPoly`](@ref)`{K,<:ConfigEnumerator}` for `K > 1`.
* Weighted graph problems is only supported for `K == 1`.

Keyword Arguments
----------------------------
* `bounded`, if it is true, use bounding trick (or boolean gradients) to reduce the working memory to store intermediate configurations.
* `tree_storage`, if it is true, it uses more memory efficient tree-structure to store the configurations.
"""
struct ConfigsMax{K, BOUNDED, TREESTORAGE} <:AbstractProperty end
ConfigsMax(K::Int=1; bounded::Bool=true, tree_storage::Bool=false) = ConfigsMax{K,bounded,tree_storage}()
max_k(::ConfigsMax{K}) where K = K
tree_storage(::ConfigsMax{K,BOUNDED,TREESTORAGE}) where {K,BOUNDED,TREESTORAGE} = TREESTORAGE

"""
    ConfigsMin{K, BOUNDED, TREESTORAGE} <:AbstractProperty
    ConfigsMin(K=1; bounded=true, tree_storage=false)

Find configurations with smallest-K sizes.

* The corresponding data type is inverted [`CountingTropical`](@ref)`{Float64,<:ConfigEnumerator}` for `K == 1` and inverted [`TruncatedPoly`](@ref)`{K,<:ConfigEnumerator}` for `K > 1`.
* Weighted graph problems is only supported for `K == 1`.

Keyword Arguments
----------------------------
* `bounded`, if it is true, use bounding trick (or boolean gradients) to reduce the working memory to store intermediate configurations.
* `tree_storage`, if it is true, it uses more memory efficient tree-structure to store the configurations.
"""
struct ConfigsMin{K, BOUNDED, TREESTORAGE} <:AbstractProperty end
ConfigsMin(K::Int=1; bounded::Bool=true, tree_storage::Bool=false) = ConfigsMin{K,bounded, tree_storage}()
min_k(::ConfigsMin{K}) where K = K
tree_storage(::ConfigsMin{K,BOUNDED,TREESTORAGE}) where {K,BOUNDED,TREESTORAGE} = TREESTORAGE

"""
    solve(problem, property; usecuda=false, T=Float64)

Solving a certain property of a graph problem.

Positional Arguments
---------------------------
* `problem` is the graph problem with tensor network information,
* `property` is string specifying the task. Using the maximum independent set problem as an example, it can be one of

    * [`SizeMax`](@ref) for finding maximum set size,
    * [`SizeMin`](@ref) for finding minimum set size,

    * [`CountingMax`](@ref) for counting configurations with largest-K sizes,
    * [`CountingMin`](@ref) for counting configurations with smallest-K sizes,
    * [`CountingAll`](@ref) for counting all configurations,
    * [`GraphPolynomial`](@ref) for evaluating the graph polynomial,

    * [`SingleConfigMax`](@ref) for finding one maximum configuration,
    * [`ConfigsMax`](@ref) for enumerating configurations with largest-K sizes,
    * [`ConfigsMin`](@ref) for enumerating configurations with smallest-K sizes,
    * [`ConfigsAll`](@ref) for enumerating all configurations,


Keyword arguments
-------------------------------------
* `usecuda` is a switch to use CUDA (if possible), user need to call statement `using CUDA` before turning on this switch.
* `T` is the "base" element type, sometimes can be used to reduce the memory cost.
"""
function solve(gp::GraphProblem, property::AbstractProperty; T=Float64, usecuda=false)
    if !_solvable(gp, property)
        throw(ArgumentError("Graph property `$(typeof(property))` is not computable for graph problem of type `$(typeof(gp))`."))
    end
    if property isa SizeMax
        return contractx(gp, _x(Tropical{T}; invert=false); usecuda=usecuda)
    elseif property isa SizeMin
        return post_invert_exponent.(contractx(gp, _x(Tropical{T}; invert=true); usecuda=usecuda))
    elseif property isa CountingAll
        return contractx(gp, one(T); usecuda=usecuda)
    elseif property isa CountingMax{1}
        return contractx(gp, _x(CountingTropical{T,T}; invert=false); usecuda=usecuda)
    elseif property isa CountingMin{1}
        return post_invert_exponent.(contractx(gp, _x(CountingTropical{T,T}; invert=true); usecuda=usecuda))
    elseif property isa CountingMax
        return contractx(gp, TruncatedPoly(ntuple(i->i == max_k(property) ? one(T) : zero(T), max_k(property)), one(T)); usecuda=usecuda)
    elseif property isa CountingMin
        return post_invert_exponent.(contractx(gp, pre_invert_exponent(TruncatedPoly(ntuple(i->i == min_k(property) ? one(T) : zero(T), min_k(property)), one(T))); usecuda=usecuda))
    elseif property isa GraphPolynomial
        return graph_polynomial(gp, Val(graph_polynomial_method(property)); usecuda=usecuda, property.kwargs...)
    elseif property isa SingleConfigMax{false}
        return solutions(gp, CountingTropical{T,T}; all=false, usecuda=usecuda, )
    elseif property isa SingleConfigMin{false}
        return solutions(gp, CountingTropical{T,T}; all=false, usecuda=usecuda, invert=true)
    elseif property isa ConfigsMax{1,false}
        return solutions(gp, CountingTropical{T,T}; all=true, usecuda=usecuda, tree_storage=tree_storage(property))
    elseif property isa ConfigsMin{1,false}
        return solutions(gp, CountingTropical{T,T}; all=true, usecuda=usecuda, invert=true, tree_storage=tree_storage(property))
    elseif property isa (ConfigsMax{K, false} where K)
        return solutions(gp, TruncatedPoly{max_k(property),T,T}; all=true, usecuda=usecuda, tree_storage=tree_storage(property))
    elseif property isa (ConfigsMin{K, false} where K)
        return solutions(gp, TruncatedPoly{min_k(property),T,T}; all=true, usecuda=usecuda, invert=true)
    elseif property isa ConfigsAll
        return solutions(gp, Real; all=true, usecuda=usecuda, tree_storage=tree_storage(property))
    elseif property isa SingleConfigMax{true}
        return best_solutions(gp; all=false, usecuda=usecuda)
    elseif property isa SingleConfigMin{true}
        return best_solutions(gp; all=false, usecuda=usecuda, invert=true)
    elseif property isa ConfigsMax{1,true}
        return best_solutions(gp; all=true, usecuda=usecuda, tree_storage=tree_storage(property))
    elseif property isa ConfigsMin{1,true}
        return best_solutions(gp; all=true, usecuda=usecuda, invert=true, tree_storage=tree_storage(property))
    elseif property isa (ConfigsMax{K,true} where K)
        return bestk_solutions(gp, max_k(property), tree_storage=tree_storage(property))
    elseif property isa (ConfigsMin{K,true} where K)
        return bestk_solutions(gp, min_k(property), invert=true, tree_storage=tree_storage(property))
    else
        error("unknown property $property.")
    end
end

# raise an error if the property for problem can not be computed
_solvable(::Any, ::Any) = true

# negate the exponents before entering the solver
pre_invert_exponent(t::TruncatedPoly{K}) where K = TruncatedPoly(t.coeffs, -t.maxorder)
pre_invert_exponent(t::TropicalNumbers.TropicalTypes) = inv(t)
# negate the exponents after entering the solver
post_invert_exponent(t::TruncatedPoly{K}) where K = TruncatedPoly(ntuple(i->t.coeffs[K-i+1], K), -t.maxorder+(K-1))
post_invert_exponent(t::TropicalNumbers.TropicalTypes) = inv(t)

"""
    max_size(problem; usecuda=false)

Returns the maximum size of the graph problem. 
A shorthand of `solve(problem, SizeMax(); usecuda=false)`.
"""
max_size(m::GraphProblem; usecuda=false) = Int(sum(solve(m, SizeMax(); usecuda=usecuda)).n)  # floating point number is faster (BLAS)

"""
    max_size_count(problem; usecuda=false)

Returns the maximum size and the counting of the graph problem.
It is a shorthand of `solve(problem, CountingMax(); usecuda=false)`.
"""
max_size_count(m::GraphProblem; usecuda=false) = (r = sum(solve(m, CountingMax(); usecuda=usecuda)); (Int(r.n), Int(r.c)))

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
