module GenericTensorNetworksCUDAExt

using CUDA
using GenericTensorNetworks
import GenericTensorNetworks: onehotmask!, togpu

# patch
# Base.ndims(::Base.Broadcast.Broadcasted{CUDA.CuArrayStyle{0}}) = 0

togpu(x::AbstractArray) = CuArray(x)

function onehotmask!(A::CuArray{T}, X::CuArray{T}) where T
    return reshape(onehotmaskvec!(vec(A), vec(X)), size(A))
end
function onehotmaskvec!(A::CuVector{T}, X::CuVector{T}) where T
    @assert length(A) == length(X)
    mask = X .≈ inv.(A)
    ci = argmax(mask)
    mask .= false
    CUDA.@allowscalar begin
        mask[ci] = true
        # set only one element in X to nonzero to help back propagating.
        xi = X[ci]
        fill!(X, zero(T))
        X[ci] = xi
    end
    return mask
end

end
