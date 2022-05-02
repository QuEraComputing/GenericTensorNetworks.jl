abstract type AbstractProperty end

"""
    SizeMax{K} <: AbstractProperty
    SizeMax(k::Int)

The maximum-K set sizes. e.g. the largest size of the [`IndependentSet`](@ref)  problem is also know as the independence number.

* The corresponding tensor element type are max-plus tropical number [`Tropical`](@ref) for `K == 1` and [`ExtendedTropical`](@ref) for `K > 1`.
* It is compatible with weighted graph problems.
* BLAS (on CPU) and GPU are supported only for `K == 1`,
"""
struct SizeMax{K} <: AbstractProperty end
SizeMax(k::Int=1) = SizeMax{k}()
max_k(::SizeMax{K}) where K = K

"""
    SizeMin{K} <: AbstractProperty
    SizeMin(k::Int)

The minimum-K set sizes. e.g. the smallest size ofthe [`MaximalIS`](@ref) problem is also known as the independent domination number.

* The corresponding tensor element type are inverted max-plus tropical number [`Tropical`](@ref) for `K == 1` and inverted [`ExtendedTropical`](@ref) for `K > 1`.
The inverted Tropical number emulates the min-plus tropical number.
* It is compatible with weighted graph problems.
* BLAS (on CPU) and GPU are supported only for `K == 1`,
"""
struct SizeMin{K} <: AbstractProperty end
SizeMin(k::Int=1) = SizeMin{k}()
max_k(::SizeMin{K}) where K = K

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
    * The corresponding tensor element type is [`Polynomial`](https://juliamath.github.io/Polynomials.jl/stable/polynomials/polynomial/#Polynomial-2).
    * It might have small round-off error depending on the data type for storing the counting.
    * It has memory overhead that linear to the graph size.
* `:fft`, use fast fourier transformation to fit the polynomial.
    * The corresponding tensor element type is `Base.Complex`.
    * It has (controllable) round-off error.
    * BLAS and GPU are supported.
    * It accepts keyword arguments `maxorder` (optional) and `r`,
        if `r > 1`, one has better precision for coefficients of large order, if `r < 1`,
        one has better precision for coefficients of small order.
* `:fitting`, fit the polynomial directly.
    * The corresponding tensor element type is floating point numbers like `Base.Float64`.
    * It has round-off error.
    * BLAS and GPU are supported, it is the fastest among all methods.

Graph polynomials are not defined for weighted graph problems.
"""
struct GraphPolynomial{METHOD} <: AbstractProperty
    kwargs
end
GraphPolynomial(; method::Symbol = :finitefield, kwargs...) = GraphPolynomial{method}(kwargs)
graph_polynomial_method(::GraphPolynomial{METHOD}) where METHOD = METHOD

"""
    SingleConfigMax{K, BOUNDED} <: AbstractProperty
    SingleConfigMax(k::Int; bounded=false)

Finding single solution for largest-K sizes, e.g. for [`IndependentSet`](@ref) problem, it is one of the maximum independent sets.

* The corresponding data type is [`CountingTropical{Float64,<:ConfigSampler}`](@ref) if `BOUNDED` is `false`, [`Tropical`](@ref) otherwise.
* Weighted graph problems is supported.
* GPU is supported,

Keyword Arguments
----------------------------
* `bounded`, if it is true, use bounding trick (or boolean gradients) to reduce the working memory to store intermediate configurations.
"""
struct SingleConfigMax{K,BOUNDED} <:AbstractProperty end
SingleConfigMax(k::Int=1; bounded::Bool=false) = SingleConfigMax{k, bounded}()
max_k(::SingleConfigMax{K}) where K = K

"""
    SingleConfigMin{K, BOUNDED} <: AbstractProperty
    SingleConfigMin(k::Int; bounded=false)

Finding single solution with smallest-K size.

* The corresponding data type is inverted [`CountingTropical{Float64,<:ConfigSampler}`](@ref) if `BOUNDED` is `false`, inverted [`Tropical`](@ref) otherwise.
* Weighted graph problems is supported.
* GPU is supported,

Keyword Arguments
----------------------------
* `bounded`, if it is true, use bounding trick (or boolean gradients) to reduce the working memory to store intermediate configurations.
"""
struct SingleConfigMin{K,BOUNDED} <:AbstractProperty end
SingleConfigMin(k::Int=1; bounded::Bool=false) = SingleConfigMin{k,bounded}()
min_k(::SingleConfigMin{K}) where K = K

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
    ConfigsMax(K=Extrema(); bounded=true, tree_storage=true)

Find configurations with largest-K sizes, e.g. for [`IndependentSet`](@ref) problem,
it is finding all independent sets of sizes ``\\alpha(G), \\alpha(G)-1, \\ldots, \\alpha(G)-K+1``.

* The corresponding data type is [`CountingTropical`](@ref)`{Float64,<:ConfigEnumerator}` for `K == Extrema()` and [`TruncatedPoly`](@ref)`{K,<:ConfigEnumerator}` for `K` being an integer.
* Weighted graph problems is only supported for `K == Extrema()`.

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

    * [`SizeMax`](@ref)`(k=1)` for finding ``k`` maximum set size,
    * [`SizeMin`](@ref)`(k=1)` for finding ``k`` minimum set size,

    * [`CountingMax`](@ref)`(k=1)` for counting configurations with maximum-``k`` sizes,
    * [`CountingMin`](@ref)`(k=1)` for counting configurations with minimum-``k`` sizes,
    * [`CountingAll`](@ref)`()` for counting all configurations,
    * [`GraphPolynomial`](@ref)`(; method=:finitefield, kwargs...)` for evaluating the graph polynomial,

    * [`SingleConfigMax`](@ref)`(k=1; bounded=false)` for finding one maximum-``k`` configuration for each size,
    * [`SingleConfigMin`](@ref)`(k=1; bounded=false)` for finding one minimum-``k`` configuration for each size,
    * [`ConfigsMax`](@ref)`(k=1; bounded=true, tree_storage=false)` for enumerating configurations with maximum-``k`` sizes,
    * [`ConfigsMin`](@ref)`(k=1; bounded=true, tree_storage=false)` for enumerating configurations with minimum-``k`` sizes,
    * [`ConfigsAll`](@ref)`(; tree_storage=false)` for enumerating all configurations,


Keyword arguments
-------------------------------------
* `usecuda` is a switch to use CUDA (if possible), user need to call statement `using CUDA` before turning on this switch.
* `T` is the "base" element type, sometimes can be used to reduce the memory cost.
"""
function solve(gp::GraphProblem, property::AbstractProperty; T=Float64, usecuda=false)
    assert_solvable(gp, property)
    if property isa SizeMax{1}
        return contractx(gp, _x(Tropical{T}; invert=false); usecuda=usecuda)
    elseif property isa SizeMin{1}
        res = contractx(gp, _x(Tropical{T}; invert=true); usecuda=usecuda)
        return asarray(post_invert_exponent.(res), res)
    elseif property isa SizeMax
        return contractx(gp, _x(ExtendedTropical{max_k(property), Tropical{T}}; invert=false); usecuda=usecuda)
    elseif property isa SizeMin
        res = contractx(gp, _x(ExtendedTropical{max_k(property), Tropical{T}}; invert=true); usecuda=usecuda)
        return asarray(post_invert_exponent.(res), res)
    elseif property isa CountingAll
        return contractx(gp, one(T); usecuda=usecuda)
    elseif property isa CountingMax{1}
        return contractx(gp, _x(CountingTropical{T,T}; invert=false); usecuda=usecuda)
    elseif property isa CountingMin{1}
        res = contractx(gp, _x(CountingTropical{T,T}; invert=true); usecuda=usecuda)
        return asarray(post_invert_exponent.(res), res)
    elseif property isa CountingMax
        return contractx(gp, TruncatedPoly(ntuple(i->i == max_k(property) ? one(T) : zero(T), max_k(property)), one(T)); usecuda=usecuda)
    elseif property isa CountingMin
        res = contractx(gp, pre_invert_exponent(TruncatedPoly(ntuple(i->i == min_k(property) ? one(T) : zero(T), min_k(property)), one(T))); usecuda=usecuda)
        return asarray(post_invert_exponent.(res), res)
    elseif property isa GraphPolynomial
        return graph_polynomial(gp, Val(graph_polynomial_method(property)); usecuda=usecuda, T=T, property.kwargs...)
    elseif property isa SingleConfigMax{1,false}
        return solutions(gp, CountingTropical{T,T}; all=false, usecuda=usecuda)
    elseif property isa (SingleConfigMax{K,false} where K)
        return solutions(gp, ExtendedTropical{max_k(property),CountingTropical{T,T}}; all=false, usecuda=usecuda)
    elseif property isa SingleConfigMin{1,false}
        return solutions(gp, CountingTropical{T,T}; all=false, usecuda=usecuda, invert=true)
    elseif property isa (SingleConfigMin{K,false} where K)
        return solutions(gp, ExtendedTropical{min_k(property),CountingTropical{T,T}}; all=false, usecuda=usecuda, invert=true)
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
    elseif property isa SingleConfigMax{1,true}
        return best_solutions(gp; all=false, usecuda=usecuda, T=T)
    elseif property isa (SingleConfigMax{K,true} where K)
        @warn "bounded `SingleConfigMax` property for `K != 1` is not implemented. Switching to the unbounded version."
        return solve(gp, SingleConfigMax{max_k(property),false}(); T, usecuda)
    elseif property isa SingleConfigMin{1,true}
        return best_solutions(gp; all=false, usecuda=usecuda, invert=true, T=T)
    elseif property isa (SingleConfigMin{K,true} where K)
        @warn "bounded `SingleConfigMin` property for `K != 1` is not implemented. Switching to the unbounded version."
        return solve(gp, SingleConfigMin{min_k(property),false}(); T, usecuda)
    elseif property isa ConfigsMax{1,true}
        return best_solutions(gp; all=true, usecuda=usecuda, tree_storage=tree_storage(property), T=T)
    elseif property isa ConfigsMin{1,true}
        return best_solutions(gp; all=true, usecuda=usecuda, invert=true, tree_storage=tree_storage(property), T=T)
    elseif property isa (ConfigsMax{K,true} where K)
        return bestk_solutions(gp, max_k(property), tree_storage=tree_storage(property), T=T)
    elseif property isa (ConfigsMin{K,true} where K)
        return bestk_solutions(gp, min_k(property), invert=true, tree_storage=tree_storage(property), T=T)
    else
        error("unknown property: `$property`.")
    end
end

# raise an error if the property for problem can not be computed
assert_solvable(::Any, ::Any) = nothing
function assert_solvable(problem, property::GraphPolynomial)
    if has_noninteger_weights(problem)
        throw(ArgumentError("Graph property `$(typeof(property))` is not computable due to having non-integer weights."))
    end
end
function assert_solvable(problem, property::ConfigsMax)
    if max_k(property) != 1 && has_noninteger_weights(problem)
        throw(ArgumentError("Graph property `$(typeof(property))` is not computable due to having non-integer weights. Maybe you wanted `SingleConfigMax($(max_k(property)))`?"))
    end
end
function assert_solvable(problem, property::ConfigsMin)
    if min_k(property) != 1 && has_noninteger_weights(problem)
        throw(ArgumentError("Graph property `$(typeof(property))` is not computable due to having non-integer weights. Maybe you wanted `SingleConfigMin($(min_k(property)))`?"))
    end
end
function assert_solvable(problem, property::CountingMax)
    if max_k(property) != 1 && has_noninteger_weights(problem)
        throw(ArgumentError("Graph property `$(typeof(property))` is not computable due to having non-integer weights. Maybe you wanted `SizeMax($(max_k(property)))`?"))
    end
end
function assert_solvable(problem, property::CountingMin)
    if min_k(property) != 1 && has_noninteger_weights(problem)
        throw(ArgumentError("Graph property `$(typeof(property))` is not computable due to having non-integer weights. Maybe you wanted `SizeMin($(min_k(property)))`?"))
    end
end
function has_noninteger_weights(problem::GraphProblem)
    for i in 1:length(terms(problem))
        if any(!isinteger, get_weights(problem, i))
            return true
        end
    end
    return false
end

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

########## memory estimation ###############
"""
    estimate_memory(problem, property; T=Float64)

Memory estimation in number of bytes to compute certain `property` of a `problem`.
`T` is the base type.
"""
function estimate_memory(problem::GraphProblem, property::AbstractProperty; T=Float64)
    _estimate_memory(tensor_element_type(T, length(labels(problem)), nflavor(problem), property), problem)
end
function estimate_memory(problem::GraphProblem, property::Union{SingleConfigMax{K,BOUNDED},SingleConfigMin{K,BOUNDED}}; T=Float64) where {K, BOUNDED}
    tc, sc, rw = timespacereadwrite_complexity(problem.code, _size_dict(problem))
    # caching all tensors is equivalent to counting the total number of writes
    if K == 1 && BOUNDED
        return ceil(Int, exp2(rw - 1)) * sizeof(Tropical{T})
    elseif K == 1 & !BOUNDED
        n, nf = length(labels(problem)), nflavor(problem)
        return peak_memory(problem.code, _size_dict(problem)) * (sizeof(tensor_element_type(T, n, nf, property)) * K)
    else
        # NOTE: the `K > 1` case does not respect bounding
        n, nf = length(labels(problem)), nflavor(problem)
        TT = tensor_element_type(T, n, nf, property)
        return peak_memory(problem.code, _size_dict(problem)) * (sizeof(tensor_element_type(T, n, nf, SingleConfigMax{1,BOUNDED}())) * K + sizeof(TT))
    end
end
function estimate_memory(problem::GraphProblem, ::GraphPolynomial{:polynomial}; T=Float64)
    # this is the upper bound
    return peak_memory(problem.code, _size_dict(problem)) * (sizeof(T) * length(labels(problem)))
end
function estimate_memory(problem::GraphProblem, ::Union{SizeMax{K},SizeMin{K}}; T=Float64) where K
    return peak_memory(problem.code, _size_dict(problem)) * (sizeof(T) * K)
end

function _size_dict(problem)
    lbs = labels(problem)
    nf = nflavor(problem)
    return Dict([lb=>nf for lb in lbs])
end

function _estimate_memory(::Type{ET}, problem::GraphProblem) where ET
    if !isbitstype(ET) && !(ET <: Mod)
        @warn "Target tensor element type `$ET` is not a bits type, the estimation of memory might be unreliable."
    end
    return peak_memory(problem.code, _size_dict(problem)) * sizeof(ET)
end

for (PROP, ET) in [
        (:(SizeMax{1}), :(Tropical{T})), (:(SizeMin{1}), :(Tropical{T})),
        (:(SizeMax{K}), :(ExtendedTropical{K,Tropical{T}})), (:(SizeMin{K}), :(ExtendedTropical{K,Tropical{T}})),
        (:(CountingAll), :T), (:(CountingMax{1}), :(CountingTropical{T,T})), (:(CountingMin{1}), :(CountingTropical{T,T})),
        (:(CountingMax{K}), :(TruncatedPoly{K,T,T})), (:(CountingMin{K}), :(TruncatedPoly{K,T,T})),
        (:(GraphPolynomial{:finitefield}), :(Mod{N,Int32} where N)), (:(GraphPolynomial{:fft}), :(Complex{T})), 
        (:(GraphPolynomial{:polynomial}), :(Polynomial{T, :x})), (:(GraphPolynomial{:fitting}), :T),
    ]
    @eval tensor_element_type(::Type{T}, n::Int, nflavor::Int, ::$PROP) where {T,K} = $ET
end

function tensor_element_type(::Type{T}, n::Int, nflavor::Int, ::PROP) where {T, K, BOUNDED, PROP<:Union{SingleConfigMax{K,BOUNDED},SingleConfigMin{K,BOUNDED}}}
    if K == 1 && BOUNDED
        return Tropical{T}
    elseif K == 1 && !BOUNDED
        return sampler_type(CountingTropical{T,T}, n, nflavor)
    else
        # NOTE: the `K > 1` case does not respect bounding
        return sampler_type(ExtendedTropical{K,CountingTropical{T,T}}, n, nflavor)
    end
end

for (PROP, ET) in [
        (:(ConfigsMax{1}), :(CountingTropical{T,T})), (:(ConfigsMin{1}), :(CountingTropical{T,T})),
        (:(ConfigsMax{K}), :(TruncatedPoly{K,T,T})), (:(ConfigsMin{K}), :(TruncatedPoly{K,T,T})),
        (:(ConfigsAll), :(Real))
    ]
    @eval function tensor_element_type(::Type{T}, n::Int, nflavor::Int, ::$PROP) where {T,K}
        set_type($ET, n, nflavor)
    end
end
