using TupleTools

"""
    backward_tropical(mode, ixs, xs, iy, y, ymask, size_dict)

The backward rule for tropical einsum.
* `mode` can be one of `:all` and `:single`,
* `ixs` and `xs` are labels and tensor data for input tensors,
* `iy` and `y` are labels and tensor data for the output tensor,
* `ymask` is the boolean mask for gradients,
* `size_dict` is a key-value map from tensor label to dimension size.
"""
function backward_tropical(mode, @nospecialize(ixs), @nospecialize(xs), @nospecialize(iy), @nospecialize(y), @nospecialize(ymask), size_dict)
    y .= inv.(y) .* ymask
    masks = []
    for i=1:length(ixs)
        nixs = TupleTools.insertat(ixs, i, (iy,))
        nxs  = TupleTools.insertat( xs, i, (y,))
        niy = ixs[i]
        if mode == :all
            mask = zeros(Bool, size(xs[i]))
            mask .= inv.(einsum(EinCode(nixs, niy), nxs, size_dict)) .== xs[i]
            push!(masks, mask)
        elseif mode == :single  # wrong, need `B` matching `A`.
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
        if X[j] == inv(A[j]) && !found
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
function cached_einsum(code::Int, @nospecialize(xs), size_dict)
    y = xs[code]
    CacheTree(y, CacheTree{eltype(y)}[])
end
function cached_einsum(code::NestedEinsum, @nospecialize(xs), size_dict)
    caches = [cached_einsum(arg, xs, size_dict) for arg in code.args]
    y = dynamic_einsum(code.eins, (getfield.(caches, :content)...,); size_info=size_dict)
    CacheTree(y, caches)
end

# computed mask tree by back propagation
function generate_masktree(code::Int, cache, mask, size_dict, mode=:all)
    CacheTree(mask, CacheTree{Bool}[])
end
function generate_masktree(code::NestedEinsum, cache, mask, size_dict, mode=:all)
    submasks = backward_tropical(mode, getixs(code.eins), (getfield.(cache.siblings, :content)...,), OMEinsum.getiy(code.eins), cache.content, mask, size_dict)
    return CacheTree(mask, generate_masktree.(code.args, cache.siblings, submasks, Ref(size_dict), mode))
end

# The masked einsum contraction
function masked_einsum(code::Int, @nospecialize(xs), masks, size_dict)
    y = copy(xs[code])
    y[OMEinsum.asarray(.!masks.content)] .= Ref(zero(eltype(y))); y
end
function masked_einsum(code::NestedEinsum, @nospecialize(xs), masks, size_dict)
    xs = [masked_einsum(arg, xs, mask, size_dict) for (arg, mask) in zip(code.args, masks.siblings)]
    y = einsum(code.eins, (xs...,), size_dict)
    y[OMEinsum.asarray(.!masks.content)] .= Ref(zero(eltype(y))); y
end

"""
    bounding_contract(code, xsa, ymask, xsb; size_info=nothing)

Contraction method with bounding.

    * `xsa` are input tensors for bounding, e.g. tropical tensors,
    * `xsb` are input tensors for computing, e.g. tensors elements are counting tropical with set algebra,
    * `ymask` is the initial gradient mask for the output tensor.
"""
function bounding_contract(@nospecialize(code::EinCode), @nospecialize(xsa), ymask, @nospecialize(xsb); size_info=nothing)
    bounding_contract(NestedEinsum((1:length(xsa)), code), xsa, ymask, xsb; size_info=size_info)
end
function bounding_contract(code::NestedEinsum, @nospecialize(xsa), ymask, @nospecialize(xsb); size_info=nothing)
    size_dict = OMEinsum.get_size_dict(collect_ixs(code), xsa, size_info)
    # compute intermediate tensors
    @debug "caching einsum..."
    c = cached_einsum(code, xsa, size_dict)
    # compute masks from cached tensors
    @debug "generating masked tree..."
    mt = generate_masktree(code, c, ymask, size_dict, :all)
    # compute results with masks
    masked_einsum(code, xsb, mt, size_dict)
end

# get the optimal solution with automatic differentiation.
function solution_ad(@nospecialize(code::EinCode), @nospecialize(xsa), ymask; size_info=nothing)
    solution_ad(NestedEinsum((1:length(xsa)), code), xsa, ymask; size_info=size_info)
end

function solution_ad(code::NestedEinsum, @nospecialize(xsa), ymask; size_info=nothing)
    size_dict = OMEinsum.get_size_dict(collect_ixs(code), xsa, size_info)
    # compute intermediate tensors
    @debug "caching einsum..."
    c = cached_einsum(code, xsa, size_dict)
    n = asscalar(c.content)
    # compute masks from cached tensors
    @debug "generating masked tree..."
    mt = generate_masktree(code, c, ymask, size_dict, :single)
    n, read_config!(code, mt, Dict())
end

function read_config!(code::NestedEinsum, mt, out)
    for (arg, ix, sibling) in zip(code.args, getixs(code.eins), mt.siblings)
        if arg isa Int
            assign = convert(Array, sibling.content)  # note: the content can be CuArray
            if length(ix) == 1
                if !assign[1] && assign[2]
                    out[ix[1]] = 1
                elseif !assign[2] && assign[1]
                    out[ix[1]] = 0
                else
                    error("invalid assign $(assign)")
                end
            end
        else  # nested
            read_config!(arg, sibling, out)
        end
    end
    return out
end