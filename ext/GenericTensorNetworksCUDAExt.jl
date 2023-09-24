module GenericTensorNetworksCUDAExt

using CUDA
using GenericTensorNetworks
import GenericTensorNetworks: onehotmask!, togpu

# patch
# Base.ndims(::Base.Broadcast.Broadcasted{CUDA.CuArrayStyle{0}}) = 0

togpu(x::AbstractArray) = CuArray(x)

function onehotmask!(A::CuArray{T}, X::CuArray{T}) where T
    @assert length(A) == length(X)
    mask = X .≈ inv.(A)
    ci = argmax(mask)
    mask .= false
    CUDA.@allowscalar mask[ci] = true
    # set some elements in X to zero to help back propagating.
    X[(!).(mask)] .= Ref(zero(T))
    return mask
end

end