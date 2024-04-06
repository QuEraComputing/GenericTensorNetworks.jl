"""
    GraphProblem

The abstract base type of graph problems.
"""
abstract type GraphProblem end
function generate_tensors(x::T, m::GraphProblem) where T
    tensors = [energy_tensors(x, m)..., extra_tensors(T, m)...]
    ixs = [energy_terms(m)..., extra_terms(m)...]
    return add_labels!(tensors, ixs, labels(m))
end
function rawcode(problem::GraphProblem; openvertices=())
    ixs = [energy_terms(problem)..., extra_terms(problem)...]
    LT = eltype(eltype(ixs))
    return DynamicEinCode(ixs, collect(LT, openvertices))  # labels for edge tensors
end

struct UnitWeight end
Base.getindex(::UnitWeight, i) = 1
Base.eltype(::UnitWeight) = Int

struct ZeroWeight end
Base.getindex(::ZeroWeight, i) = 0
Base.eltype(::ZeroWeight) = Int

"""
$TYPEDEF
    GenericTensorNetwork(problem::GraphProblem; openvertices=(), fixedvertices=Dict(), optimizer=GreedyMethod())

The generic tensor network that generated from a [`GraphProblem`](@ref).

Positional arguments
-------------------------------
* `problem` is the graph problem.
* `code` is the tensor network contraction code.
* `fixedvertices` is a dictionary specifying the fixed dimensions.
"""
struct GenericTensorNetwork{CFG, CT, LT}
    problem::CFG
    code::CT
    fixedvertices::Dict{LT,Int}
end
function GenericTensorNetwork(problem::GraphProblem; openvertices=(), fixedvertices=Dict(), optimizer=GreedyMethod())
    rcode = rawcode(problem; openvertices)
    code = _optimize_code(rcode, uniformsize_fix(rcode, nflavor(problem), fixedvertices), optimizer, MergeVectors())
    return GenericTensorNetwork(problem, code, Dict{labeltype(code),Int}(fixedvertices))
end
function generate_tensors(x::T, tn::GenericTensorNetwork) where {T}
    ixs = getixsv(tn.code)
    isempty(ixs) && return Array{T}[]
    tensors = generate_tensors(x, tn.problem)
    return select_dims(tensors, ixs, fixedvertices(tn))
end

######## Interfaces for graph problems ##########
"""
    get_weights(problem::GraphProblem[, i::Int]) -> Vector
    get_weights(problem::GenericTensorNetwork[, i::Int]) -> Vector

The weights for the `problem` or the weights for the degree of freedom specified by the `i`-th term if a second argument is provided.
Weights are associated with [`energy_terms`](@ref) in the graph problem.
In graph polynomial, integer weights can be the exponents.
"""
function get_weights end
get_weights(gp::GenericTensorNetwork) = get_weights(gp.problem)
get_weights(gp::GenericTensorNetwork, i::Int) = get_weights(gp.problem, i)

"""
    chweights(problem::GraphProblem, weights) -> GraphProblem
    chweights(problem::GenericTensorNetwork, weights) -> GenericTensorNetwork

Change the weights for the `problem` and return a new problem instance.
Weights are associated with [`energy_terms`](@ref) in the graph problem.
In graph polynomial, integer weights can be the exponents.
"""
function chweights end
chweights(gp::GenericTensorNetwork, weights) = GenericTensorNetwork(chweights(gp.problem, weights), gp.code, gp.fixedvertices)

"""
    labels(problem::GraphProblem) -> Vector
    labels(problem::GenericTensorNetwork) -> Vector

The labels of a graph problem is defined as the degrees of freedoms in the graph problem.
e.g. for the maximum independent set problems, they are the indices of vertices: 1, 2, 3...,
while for the max cut problem, they are the edges.
"""
labels(gp::GenericTensorNetwork) = labels(gp.problem)

"""
    energy_terms(problem::GraphProblem) -> Vector
    energy_terms(problem::GenericTensorNetwork) -> Vector

The energy terms of a graph problem is defined as the tensor labels that carrying local energies (or weights) in the graph problem.
"""
function energy_terms end
energy_terms(gp::GenericTensorNetwork) = energy_terms(gp.problem)

"""
    extra_terms(problem::GraphProblem) -> Vector
    extra_terms(problem::GenericTensorNetwork) -> Vector

The extra terms of a graph problem is defined as the tensor labels that not carrying local energies (or weights) in the graph problem.
"""
function extra_terms end
extra_terms(gp::GenericTensorNetwork) = extra_terms(gp.problem)

"""
    fixedvertices(tnet::GenericTensorNetwork) -> Dict

Returns the fixed vertices in the graph problem, which is a dictionary specifying the fixed dimensions.
"""
fixedvertices(gtn::GenericTensorNetwork) = gtn.fixedvertices

"""
    flavors(::Type{<:GraphProblem}) -> Vector
    flavors(::Type{<:GenericTensorNetwork}) -> Vector

It returns a vector of integers as the flavors of a degree of freedom.
Its size is the same as the degree of freedom on a single vertex/edge.
"""
flavors(::GT) where GT<:GraphProblem = flavors(GT)
flavors(::GenericTensorNetwork{GT}) where GT<:GraphProblem = flavors(GT)

"""
    nflavor(::Type{<:GraphProblem}) -> Int
    nflavor(::Type{<:GenericTensorNetwork}) -> Int
    nflavor(::GT) where GT<:GraphProblem -> Int
    nflavor(::GenericTensorNetwork{GT}) where GT<:GraphProblem -> Int

Bond size is equal to the number of flavors.
"""
nflavor(::Type{GT}) where GT = length(flavors(GT))
nflavor(::Type{GenericTensorNetwork{GT}}) where GT = nflavor(GT)
nflavor(::GT) where GT<:GraphProblem = nflavor(GT)
nflavor(::GenericTensorNetwork{GT}) where GT<:GraphProblem = nflavor(GT)

"""
    generate_tensors(func, problem::GenericTensorNetwork)

Generate a vector of tensors as the inputs of the tensor network contraction code `problem.code`.
`func` is a function to customize the tensors.
`func(symbol)` returns a vector of elements, the length of which is same as the number of flavors.

Example
--------------------------
The following code gives your the maximum independent set size
```jldoctest
julia> using Graphs, GenericTensorNetworks

julia> gp = GenericTensorNetwork(IndependentSet(smallgraph(:petersen)));

julia> getixsv(gp.code)
25-element Vector{Vector{Int64}}:
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]
 [10]
 ⋮
 [3, 8]
 [4, 5]
 [4, 9]
 [5, 10]
 [6, 8]
 [6, 9]
 [7, 9]
 [7, 10]
 [8, 10]

julia> gp.code(GenericTensorNetworks.generate_tensors(Tropical(1.0), gp)...)
0-dimensional Array{Tropical{Float64}, 0}:
4.0ₜ
```
"""
function generate_tensors end

# requires field `code`

include("IndependentSet.jl")
include("MaximalIS.jl")
include("MaxCut.jl")
include("Matching.jl")
include("Coloring.jl")
include("PaintShop.jl")
include("Satisfiability.jl")
include("DominatingSet.jl")
include("SetPacking.jl")
include("SetCovering.jl")
include("OpenPitMining.jl")
include("SpinGlass.jl")

# forward the time, space and readwrite complexity
OMEinsum.contraction_complexity(gp::GenericTensorNetwork) = contraction_complexity(gp.code, uniformsize(gp.code, nflavor(gp)))
# the following two interfaces will be deprecated
OMEinsum.timespace_complexity(gp::GenericTensorNetwork) = timespace_complexity(gp.code, uniformsize(gp.code, nflavor(gp)))
OMEinsum.timespacereadwrite_complexity(gp::GenericTensorNetwork) = timespacereadwrite_complexity(gp.code, uniformsize(gp.code, nflavor(gp)))

# contract the graph tensor network
function contractx(gp::GenericTensorNetwork, x; usecuda=false)
    @debug "generating tensors for x = `$x` ..."
    xs = generate_tensors(x, gp)
    @debug "contracting tensors ..."
    if usecuda
        gp.code([togpu(x) for x in xs]...)
    else
        gp.code(xs...)
    end
end

function uniformsize_fix(code, dim, fixedvertices)
    size_dict = uniformsize(code, dim)
    for key in keys(fixedvertices)
        size_dict[key] = 1
    end
    return size_dict
end

# multiply labels vectors to the generate tensor.
add_labels!(tensors::AbstractVector{<:AbstractArray}, ixs, labels) = tensors

const SetPolyNumbers{T} = Union{Polynomial{T}, TruncatedPoly{K,T} where K, CountingTropical{TV,T} where TV} where T<:AbstractSetNumber
function add_labels!(tensors::AbstractVector{<:AbstractArray{T}}, ixs, labels) where T <: Union{AbstractSetNumber, SetPolyNumbers, ExtendedTropical{K,T} where {K,T<:SetPolyNumbers}}
    for (t, ix) in zip(tensors, ixs)
        for (dim, l) in enumerate(ix)
            index = findfirst(==(l), labels)
            v = [_onehotv(T, index, k-1) for k=1:size(t, dim)]
            t .*= reshape(v, ntuple(j->dim==j ? length(v) : 1, ndims(t)))
        end
    end
    return tensors
end

# select dimensions from tensors
# `tensors` is a vector of tensors,
# `ixs` is the tensor labels for `tensors`.
# `fixedvertices` is a dictionary specifying the fixed dimensions, check [`fixedvertices`](@ref)
function select_dims(tensors::AbstractVector{<:AbstractArray{T}}, ixs, fixedvertices::AbstractDict) where T
    isempty(fixedvertices) && return tensors
    map(tensors, ixs) do t, ix
        dims = map(ixi->ixi ∉ keys(fixedvertices) ? Colon() : (fixedvertices[ixi]+1:fixedvertices[ixi]+1), ix)
        t[dims...]
    end
end

_pow(x, i) = x^i
function _pow(x::LaurentPolynomial{BS,X}, i) where {BS,X}
    if i >= 0
        return x^i
    else
        @assert length(x.coeffs) == 1
        return LaurentPolynomial(x.coeffs .^ i, x.order[]*i)
    end
end

# TODOs:
# 1. Dominating set
# \exists x_i,\ldots,x_K \forall y\left[\bigwedge_{i=1}^{K}(y=x_i\wedge \textbf{adj}(y, x_i))\right]
# 2. Polish reading data
#     * consistent configuration assign of max-cut
# 3. Support transverse field in max-cut
