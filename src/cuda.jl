using .CUDA

function onehotmask(A::CuArray{T}, X::CuArray{T}) where T
    mask = X .== inv.(A)
    ci = argmax(mask)
    mask .= false
    mask[CuArray([ci])] = true
    return mask
end