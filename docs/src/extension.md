# Make extensions

The code we will show below is a gist of how this package is implemented for pedagogical purpose, which covers many functionalities of the main repo without caring much about performance.
This project depends on multiple open source packages in the Julia ecosystem:

* [OMEinsum](https://github.com/under-Peter/OMEinsum.jl) and [OMEinsumContractionOrders](https://github.com/TensorBFS/OMEinsumContractionOrders.jl) are packages providing the support for Einstein's (or tensor network) notation and state-of-the-art algorithms for contraction order optimization, which includes multiple state of the art algorithms.
* [TropicalNumbers](https://github.com/TensorBFS/TropicalNumbers.jl) and [TropicalGEMM](https://github.com/TensorBFS/TropicalGEMM.jl) are packages providing tropical number and efficient tropical matrix multiplication.
* [Graphs](https://github.com/JuliaGraphs/Graphs.jl) is a foundational package for graph manipulation in the Julia community.
* [Polynomials](https://github.com/JuliaMath/Polynomials.jl) is a package providing polynomial algebra and polynomial fitting.
* [Mods](https://github.com/scheinerman/Mods.jl) and the [Primes](https://github.com/JuliaMath/Primes.jl) package providing finite field algebra and prime number manipulation.

They can be installed in a similar way to `GenericTensorNetworks`.
After installing the required packages, one can open a Julia REPL, and copy-paste the following code snippet into it.

```julia
using OMEinsum, OMEinsumContractionOrders
using Graphs
using Random

# generate a random regular graph of size 50, degree 3
graph = (Random.seed!(2); Graphs.random_regular_graph(50, 3))

# generate einsum code, i.e. the labels of tensors
code = EinCode(([minmax(e.src,e.dst) for e in Graphs.edges(graph)]..., # labels for edge tensors
                [(i,) for i in Graphs.vertices(graph)]...), ())        # labels for vertex tensors

# an einsum contraction without a contraction order specified is called `EinCode`,
# an einsum contraction having a contraction order (specified as a tree structure) is called `NestedEinsum`.
# assign each label a dimension-2, it will be used in the contraction order optimization
# `uniquelabels` function extracts the tensor labels into a vector.
size_dict = Dict([s=>2 for s in uniquelabels(code)])
# optimize the contraction order using the `TreeSA` method; the target space complexity is 2^17
optimized_code = optimize_code(code, size_dict, TreeSA())
println("time/space complexity is $(OMEinsum.timespace_complexity(optimized_code, size_dict))")

# a function for computing the independence polynomial
function independence_polynomial(x::T, code) where {T}
	xs = map(getixsv(code)) do ix
        # if the tensor rank is 1, create a vertex tensor.
        # otherwise the tensor rank must be 2, create a bond tensor.
        length(ix)==1 ? [one(T), x] : [one(T) one(T); one(T) zero(T)]
    end
    # both `EinCode` and `NestedEinsum` are callable, inputs are tensors.
	code(xs...)
end

########## COMPUTING THE MAXIMUM INDEPENDENT SET SIZE AND ITS COUNTING/DEGENERACY ###########

# using Tropical numbers to compute the MIS size and the MIS degeneracy.
using TropicalNumbers
mis_size(code) = independence_polynomial(TropicalF64(1.0), code)[]
println("the maximum independent set size is $(mis_size(optimized_code).n)")

# A `CountingTropical` object has two fields, tropical field `n` and counting field `c`.
mis_count(code) = independence_polynomial(CountingTropical{Float64,Float64}(1.0, 1.0), code)[]
println("the degeneracy of maximum independent sets is $(mis_count(optimized_code).c)")

########## COMPUTING THE INDEPENDENCE POLYNOMIAL ###########

# using Polynomial numbers to compute the polynomial directly
using Polynomials
println("the independence polynomial is $(independence_polynomial(Polynomial([0.0, 1.0]), optimized_code)[])")

########## FINDING MIS CONFIGURATIONS ###########

# define the set algebra
struct ConfigEnumerator{N}
    # NOTE: BitVector is dynamic and it can be very slow; check our repo for the static version
    data::Vector{BitVector}
end
function Base.:+(x::ConfigEnumerator{N}, y::ConfigEnumerator{N}) where {N}
    res = ConfigEnumerator{N}(vcat(x.data, y.data))
    return res
end
function Base.:*(x::ConfigEnumerator{L}, y::ConfigEnumerator{L}) where {L}
    M, N = length(x.data), length(y.data)
    z = Vector{BitVector}(undef, M*N)
    for j=1:N, i=1:M
        z[(j-1)*M+i] = x.data[i] .| y.data[j]
    end
    return ConfigEnumerator{L}(z)
end
Base.zero(::Type{ConfigEnumerator{N}}) where {N} = ConfigEnumerator{N}(BitVector[])
Base.one(::Type{ConfigEnumerator{N}}) where {N} = ConfigEnumerator{N}([falses(N)])

# the algebra sampling one of the configurations
struct ConfigSampler{N}
    data::BitVector
end

function Base.:+(x::ConfigSampler{N}, y::ConfigSampler{N}) where {N}  # biased sampling: return `x`
    return x  # randomly pick one
end
function Base.:*(x::ConfigSampler{L}, y::ConfigSampler{L}) where {L}
    ConfigSampler{L}(x.data .| y.data)
end

Base.zero(::Type{ConfigSampler{N}}) where {N} = ConfigSampler{N}(trues(N))
Base.one(::Type{ConfigSampler{N}}) where {N} = ConfigSampler{N}(falses(N))

# enumerate all configurations if `all` is true; compute one otherwise.
# a configuration is stored in the data type of `StaticBitVector`; it uses integers to represent bit strings.
# `ConfigTropical` is defined in `TropicalNumbers`. It has two fields: tropical number `n` and optimal configuration `config`.
# `CountingTropical{T,<:ConfigEnumerator}` stores configurations instead of simple counting.
function mis_config(code; all=false)
    # map a vertex label to an integer
    vertex_index = Dict([s=>i for (i, s) in enumerate(uniquelabels(code))])
    N = length(vertex_index)  # number of vertices
    xs = map(getixsv(code)) do ix
        T = all ? CountingTropical{Float64, ConfigEnumerator{N}} : CountingTropical{Float64, ConfigSampler{N}}
        if length(ix) == 2
            return [one(T) one(T); one(T) zero(T)]
        else
            s = falses(N)
            s[vertex_index[ix[1]]] = true  # one hot vector
            if all
                [one(T), T(1.0, ConfigEnumerator{N}([s]))]
            else
                [one(T), T(1.0, ConfigSampler{N}(s))]
            end
        end
    end
	return code(xs...)
end

println("one of the optimal configurations is $(mis_config(optimized_code; all=false)[].c.data)")

# direct enumeration of configurations can be very slow; please check the bounding version in our Github repo.
println("all optimal configurations are $(mis_config(optimized_code; all=true)[].c)")
```
