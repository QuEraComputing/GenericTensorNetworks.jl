function generate_tensors(x::T, m::ConstraintSatisfactionProblem) where T
    terms = ProblemReductions.size_terms(m)
    tensors = [reshape(map(s -> !s.is_valid ? zero(x) : _pow(x, s.size), t.solution_sizes), ntuple(i->num_flavors(m), length(t.variables))) for t in terms]
    ixs = [t.variables for t in terms]
    return add_labels!(tensors, ixs, variables(m))
end
function rawcode(problem::ConstraintSatisfactionProblem; openvertices=())
    ixs = [t.variables for t in ProblemReductions.size_terms(problem)]
    LT = eltype(eltype(ixs))
    return DynamicEinCode(ixs, collect(LT, openvertices))  # labels for edge tensors
end

"""
$TYPEDEF
    GenericTensorNetwork(problem::ConstraintSatisfactionProblem; openvertices=(), fixedvertices=Dict(), optimizer=GreedyMethod())

The generic tensor network that generated from a [`ConstraintSatisfactionProblem`](@ref).

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
function GenericTensorNetwork(problem::ConstraintSatisfactionProblem; openvertices=(), fixedvertices=Dict(), optimizer=GreedyMethod())
    rcode = rawcode(problem; openvertices)
    code = _optimize_code(rcode, uniformsize_fix(rcode, num_flavors(problem), fixedvertices), optimizer, MergeVectors())
    return GenericTensorNetwork(problem, code, Dict{labeltype(code),Int}(fixedvertices))
end
function Base.show(io::IO, tn::GenericTensorNetwork)
    println(io, "$(typeof(tn))")
    println(io, "- open vertices: $(getiyv(tn.code))")
    println(io, "- fixed vertices: $(tn.fixedvertices)")
    tc, sc, rw = contraction_complexity(tn)
    print(io, "- contraction time = 2^$(round(tc; digits=3)), space = 2^$(round(sc; digits=3)), read-write = 2^$(round(rw; digits=3))")
end
Base.show(io::IO, ::MIME"text/plain", tn::GenericTensorNetwork) = Base.show(io, tn)
function generate_tensors(x::T, tn::GenericTensorNetwork) where {T}
    ixs = getixsv(tn.code)
    isempty(ixs) && return Array{T}[]
    tensors = generate_tensors(x, tn.problem)
    return select_dims(tensors, ixs, fixedvertices(tn))
end

variables(gp::GenericTensorNetwork) = variables(gp.problem)
set_weights(gp::GenericTensorNetwork, weights) = GenericTensorNetwork(set_weights(gp.problem, weights), gp.code, gp.fixedvertices)
weights(gp::GenericTensorNetwork) = weights(gp.problem)

"""
    fixedvertices(tnet::GenericTensorNetwork) -> Dict

Returns the fixed vertices in the graph problem, which is a dictionary specifying the fixed dimensions.
"""
fixedvertices(gtn::GenericTensorNetwork) = gtn.fixedvertices

"""
    flavors(::Type{<:GenericTensorNetwork}) -> Vector

It returns a vector of integers as the flavors of a degree of freedom.
Its size is the same as the degree of freedom on a single vertex/edge.
"""
flavors(::GenericTensorNetwork{GT}) where GT<:ConstraintSatisfactionProblem = flavors(GT)

"""
    num_flavors(::GenericTensorNetwork{GT}) where GT<:ConstraintSatisfactionProblem -> Int

Bond size is equal to the number of flavors.
"""
num_flavors(::GenericTensorNetwork{GT}) where GT<:ConstraintSatisfactionProblem = num_flavors(GT)

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
 [1, 2]
 [1, 5]
 [1, 6]
 [2, 3]
 [2, 7]
 [3, 4]
 [3, 8]
 [4, 5]
 [4, 9]
 [5, 10]
 ⋮
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]
 [10]

julia> gp.code(GenericTensorNetworks.generate_tensors(Tropical(1.0), gp)...)
0-dimensional Array{Tropical{Float64}, 0}:
4.0ₜ
```
"""
function generate_tensors end

# forward the time, space and readwrite complexity
OMEinsum.contraction_complexity(gp::GenericTensorNetwork) = contraction_complexity(gp.code, uniformsize(gp.code, num_flavors(gp)))
# the following two interfaces will be deprecated
OMEinsum.timespace_complexity(gp::GenericTensorNetwork) = timespace_complexity(gp.code, uniformsize(gp.code, num_flavors(gp)))
OMEinsum.timespacereadwrite_complexity(gp::GenericTensorNetwork) = timespacereadwrite_complexity(gp.code, uniformsize(gp.code, num_flavors(gp)))

# contract the graph tensor network
function contractx(gp::GenericTensorNetwork, x; usecuda=false)
    @debug "generating tensors for x = `$x` ..."
    xs = generate_tensors(x, gp)
    length(xs) == 0 && return asarray(one(x))  # empty tensor network
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

####### Extra utilities #######
"""
    mis_compactify!(tropicaltensor; potential=nothing)

Compactify tropical tensor for maximum independent set problem. It will eliminate
some entries by setting them to zero, by the criteria that removing these entry
does not change the MIS size of its parent graph (reference to be added).

### Arguments
- `tropicaltensor::AbstractArray{T}`: the tropical tensor

### Keyword arguments
- `potential=nothing`: the maximum possible MIS contribution from each open vertex
"""
function mis_compactify!(a::AbstractArray{T, N}; potential=nothing) where {T <: TropicalTypes, N}
    @assert potential === nothing || length(potential) == N "got unexpected potential length: $(length(potential)), expected $N"
	for (ind_a, val_a) in enumerate(a)
		for (ind_b, val_b) in enumerate(a)
			bs_a = ind_a - 1
			bs_b = ind_b - 1
            if worse_than(bs_a, bs_b, val_a.n, val_b.n, potential)
                @inbounds a[ind_a] = zero(T)
            end
		end
	end
	return a
end
function worse_than(bs_a::Integer, bs_b::Integer, val_a::T, val_b::T, potential::AbstractVector) where T
    bs_a != bs_b && val_a + sum(k->readbit(bs_a, k) < readbit(bs_b, k) ? potential[k] : zero(T), 1:length(potential)) <= val_b
end
function worse_than(bs_a::Integer, bs_b::Integer, val_a::T, val_b::T, ::Nothing) where T
    bs_a != bs_b && val_a <= val_b && (bs_b & bs_a) == bs_b
end
readbit(bs::Integer, k::Integer) = (bs >> (k-1)) & 1

