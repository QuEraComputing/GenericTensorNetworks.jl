using .CUDA
using TropicalGEMM: XTranspose, NativeTypes, Tropical
using LinearAlgebra

function onehotmask(A::CuArray{T}, X::CuArray{T}) where T
    mask = X .== inv.(A)
    ci = argmax(mask)
    mask .= false
    mask[CuArray([ci])] = true
    return mask
end

# fix the matrix multiplication ambiguity
const CTranspose{T} = Transpose{T, <:StridedCuVecOrMat}
for (TA, CTA) in [(:AbstractMatrix, :CuMatrix), (:XTranspose, :CTranspose)]
    for (TB, CTB) in [(:AbstractMatrix, :CuMatrix), (:XTranspose, :CTranspose)]
        @eval function LinearAlgebra.mul!(o::CuMatrix{T}, a::$TA{T}, b::$TB{T}, α::Number, β::Number) where {T<:Tropical{<:NativeTypes}}
            invoke(LinearAlgebra.mul!, Tuple{CuMatrix, $CTA, $CTB, Number, Number}, o, a, b, α, β)
        end
    end
end