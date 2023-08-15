using .CUDA
using SIMDTypes: NativeTypes  # avoid ambiguity error when using `TropicalGEMM`
using TropicalNumbers: Tropical, TropicalTypes
using LinearAlgebra

# patch
Base.ndims(::Base.Broadcast.Broadcasted{CUDA.CuArrayStyle{0}}) = 0

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

# fix the matrix multiplication ambiguity
const CTranspose{T} = Transpose{T, <:StridedCuVecOrMat{T}}
for TT in [:(Tropical{<:NativeTypes}), :TropicalTypes]
   for RT in [TT, :Real]
       for (TA, CTA, tA) in [(:CuMatrix, :CuMatrix, 'N'), (:CTranspose, :(Transpose{<:Any, <:StridedCuVecOrMat}), 'T')]
           for (TB, CTB, tB) in [(:CuMatrix, :CuMatrix, 'N'), (:CTranspose, :(Transpose{<:Any, <:StridedCuVecOrMat}), 'T')]
               @eval function LinearAlgebra.mul!(o::CuMatrix{T}, a::$TA{T}, b::$TB{T}, α::$RT, β::$RT) where {T<:$TT}
                   LinearAlgebra.generic_matmatmul!(o, $tA, $tB, parent(a), parent(b), LinearAlgebra.MulAddMul(α, β))
               end
           end
       end
   end
end
