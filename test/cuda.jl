using CUDA, Random
using LinearAlgebra: mul!
using GraphTensorNetworks, Test

@testset "cuda patch" begin
    for T in [Tropical{Float64}, CountingTropical{Float64,Float64}]
        a = T.(CUDA.randn(4, 4))
        b = T.(CUDA.randn(4))
        for A in [transpose(a), a, transpose(b)]
            for B in [transpose(a), a, b]
                if !(size(A) == (1,4) && size(B) == (4,))
                    res0 = Array(A) * Array(B)
                    res1 = A * B
                    res2 = mul!(CUDA.zeros(T, size(res0)...), A, B, true, false)
                    @test Array(res1) ≈ res0
                    @test Array(res2) ≈ res0
                end
            end
        end
    end
end