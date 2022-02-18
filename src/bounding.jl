using OMEinsum: DynamicEinCode

struct AllConfigs{K} end
largest_k(::AllConfigs{K}) where K = K
struct SingleConfig end

"""
    backward_tropical(mode, ixs, xs, iy, y, ymask, size_dict)

The backward rule for tropical einsum.

* `mode` can be one of `:all` and `:single`,
* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `ymask` is the boolean mask for gradients,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_tropical(mode, ixs, @nospecialize(xs::Tuple), iy, @nospecialize(y), @nospecialize(ymask), size_dict)
    y .= inv.(y) .* ymask
    masks = []
    for i=1:length(ixs)
        nixs = OMEinsum._insertat(ixs, i, iy)
        nxs  = OMEinsum._insertat( xs, i, y)
        niy = ixs[i]
        if mode isa AllConfigs
            mask = zeros(Bool, size(xs[i]))
            # here, we set a threshold `1e-12` to avoid round off errors.
            mask .= inv.(einsum(EinCode(nixs, niy), nxs, size_dict)) .<= xs[i] .* Tropical(largest_k(mode)-1+1e-12)
            push!(masks, mask)
        elseif mode isa SingleConfig
            A = zeros(eltype(xs[i]), size(xs[i]))
            A = einsum(EinCode(nixs, niy), nxs, size_dict)
            push!(masks, onehotmask(A, xs[i]))
        else
            error("unkown mode: $mod")
        end
    end
    return masks
end

# one of the entry in `A` that equal to the corresponding entry in `X` is masked to true.
function onehotmask(A::AbstractArray{T}, X::AbstractArray{T}) where T
    @assert length(A) == length(X)
    mask = falses(size(A)...)
    found = false
    @inbounds for j=1:length(A)
        if X[j] ≈ inv(A[j]) && !found
            mask[j] = true
            found = true
        else
            X[j] = zero(T)
        end
    end
    return mask
end

# the data structure storing intermediate `NestedEinsum` contraction results.
struct CacheTree{T}
    content::AbstractArray{T}
    siblings::Vector{CacheTree{T}}
end
function cached_einsum(se::SlicedEinsum, @nospecialize(xs), size_dict)
    if length(se.slicing) != 0
        @warn "Slicing is not supported for caching! Fallback to `NestedEinsum`."
    end
    return cached_einsum(se.eins, xs, size_dict)
end
function cached_einsum(code::NestedEinsum, @nospecialize(xs), size_dict)
    if OMEinsum.isleaf(code)
        y = xs[code.tensorindex]
        return CacheTree(y, CacheTree{eltype(y)}[])
    else
        caches = [cached_einsum(arg, xs, size_dict) for arg in code.args]
        y = einsum(code.eins, ntuple(i->caches[i].content, length(caches)), size_dict)
        return CacheTree(y, caches)
    end
end

# computed mask tree by back propagation
function generate_masktree(mode, se::SlicedEinsum, cache, mask, size_dict)
    if length(se.slicing) != 0
        @warn "Slicing is not supported for generating masked tree! Fallback to `NestedEinsum`."
    end
    return generate_masktree(mode, se.eins, cache, mask, size_dict)
end
function generate_masktree(mode, code::NestedEinsum, cache, mask, size_dict)
    if OMEinsum.isleaf(code)
        return CacheTree(mask, CacheTree{Bool}[])
    else
        submasks = backward_tropical(mode, getixs(code.eins), (getfield.(cache.siblings, :content)...,), OMEinsum.getiy(code.eins), cache.content, mask, size_dict)
        return CacheTree(mask, generate_masktree.(Ref(mode), code.args, cache.siblings, submasks, Ref(size_dict)))
    end
end

# The masked einsum contraction
function masked_einsum(se::SlicedEinsum, @nospecialize(xs), masks, size_dict)
    if length(se.slicing) != 0
        @warn "Slicing is not supported for masked contraction! Fallback to `NestedEinsum`."
    end
    return masked_einsum(se.eins, xs, masks, size_dict)
end
function masked_einsum(code::NestedEinsum, @nospecialize(xs), masks, size_dict)
    if OMEinsum.isleaf(code)
        y = copy(xs[code.tensorindex])
        y[OMEinsum.asarray(.!masks.content)] .= Ref(zero(eltype(y)))
        return y
    else
        xs = [masked_einsum(arg, xs, mask, size_dict) for (arg, mask) in zip(code.args, masks.siblings)]
        y = einsum(code.eins, (xs...,), size_dict)
        y[OMEinsum.asarray(.!masks.content)] .= Ref(zero(eltype(y)))
        return y
    end
end

"""
    bounding_contract(mode, code, xsa, ymask, xsb; size_info=nothing)

Contraction method with bounding.

* `mode` is a `AllConfigs{K}` instance, where `MIS-K+1` is the largest IS size that you care about.
* `xsa` are input tensors for bounding, e.g. tropical tensors,
* `xsb` are input tensors for computing, e.g. tensors elements are counting tropical with set algebra,
* `ymask` is the initial gradient mask for the output tensor.
"""
function bounding_contract(mode::AllConfigs, code::EinCode, @nospecialize(xsa), ymask, @nospecialize(xsb); size_info=nothing)
    LT = OMEinsum.labeltype(code)
    bounding_contract(mode, NestedEinsum(NestedEinsum{DynamicEinCode{LT}}.(1:length(xsa)), code), xsa, ymask, xsb; size_info=size_info)
end
function bounding_contract(mode::AllConfigs, code::Union{NestedEinsum,SlicedEinsum}, @nospecialize(xsa), ymask, @nospecialize(xsb); size_info=nothing)
    size_dict = size_info===nothing ? Dict{OMEinsum.labeltype(code.eins),Int}() : copy(size_info)
    OMEinsum.get_size_dict!(code, xsa, size_dict)
    # compute intermediate tensors
    @debug "caching einsum..."
    c = cached_einsum(code, xsa, size_dict)
    # compute masks from cached tensors
    @debug "generating masked tree..."
    mt = generate_masktree(mode, code, c, ymask, size_dict)
    # compute results with masks
    masked_einsum(code, xsb, mt, size_dict)
end

# get the optimal solution with automatic differentiation.
function solution_ad(code::EinCode, @nospecialize(xsa), ymask; size_info=nothing)
    LT = OMEinsum.labeltype(code)
    solution_ad(NestedEinsum(NestedEinsum{DynamicEinCode{LT}}.(1:length(xsa)), code), xsa, ymask; size_info=size_info)
end

function solution_ad(code::Union{NestedEinsum,SlicedEinsum}, @nospecialize(xsa), ymask; size_info=nothing)
    size_dict = size_info===nothing ? Dict{OMEinsum.labeltype(code.eins),Int}() : copy(size_info)
    OMEinsum.get_size_dict!(code, xsa, size_dict)
    # compute intermediate tensors
    @debug "caching einsum..."
    c = cached_einsum(code, xsa, size_dict)
    n = asscalar(c.content)
    # compute masks from cached tensors
    @debug "generating masked tree..."
    mt = generate_masktree(SingleConfig(), code, c, ymask, size_dict)
    config = read_config!(code, mt, Dict())
    if length(config) !== length(labels(code))  # equal to the # of degree of freedoms
        error("configuration `$(config)` is not fully determined!")
    end
    n, config
end

# get the solution configuration from gradients.
function read_config!(code::SlicedEinsum, mt, out)
    read_config!(code.eins, mt, out)
end

function read_config!(code::NestedEinsum, mt, out)
    for (arg, ix, sibling) in zip(code.args, getixs(code.eins), mt.siblings)
        if OMEinsum.isleaf(arg)
            mask = convert(Array, sibling.content)  # note: the content can be CuArray
            for ci in CartesianIndices(mask)
                if mask[ci]
                    for k in 1:ndims(mask)
                        if haskey(out, ix[k])
                            @assert out[ix[k]] == ci.I[k] - 1
                        else
                            out[ix[k]] = ci.I[k] - 1
                        end
                    end
                end
            end
        else  # nested
            read_config!(arg, sibling, out)
        end
    end
    return out
end