using .CUDA
using SIMDTypes: NativeTypes  # avoid ambiguity error when using `TropicalGEMM`
using TropicalNumbers: Tropical, TropicalTypes
using LinearAlgebra

# patch
Base.ndims(::Base.Broadcast.Broadcasted{CUDA.CuArrayStyle{0}}) = 0

function onehotmask(A::CuArray{T}, X::CuArray{T}) where T
    mask = X .== inv.(A)
    ci = argmax(mask)
    mask .= false
    mask[CuArray([ci])] = true
    return mask
end

# fix the matrix multiplication ambiguity
const CTranspose{T} = Transpose{T, <:StridedCuVecOrMat{T}}
for TT in [:(Tropical{<:NativeTypes}), :TropicalTypes]
   for RT in [TT, :Real]
       for (TA, CTA) in [(:CuMatrix, :CuMatrix), (:CTranspose, :(Transpose{<:Any, <:StridedCuVecOrMat}))]
           for (TB, CTB) in [(:CuMatrix, :CuMatrix), (:CTranspose, :(Transpose{<:Any, <:StridedCuVecOrMat}))]
               @eval function LinearAlgebra.mul!(o::CuMatrix{T}, a::$TA{T}, b::$TB{T}, α::$RT, β::$RT) where {T<:$TT}
                   CUDA.CUBLAS.gemm_dispatch!(o, a, b, α, β)
               end
           end
       end
   end
end
