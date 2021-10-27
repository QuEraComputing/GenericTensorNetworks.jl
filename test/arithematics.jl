using GraphTensorNetworks, Test, OMEinsum, OMEinsumContractionOrders
using Mods, Polynomials, TropicalNumbers
using Graphs, Random
using GraphTensorNetworks: StaticBitVector

@testset "truncated poly" begin
    p1 = TruncatedPoly((2,2,1), 2.0)
    p2 = TruncatedPoly((2,3,9), 4.0)
    x = Polynomial([2, 2, 1])
    y = Polynomial([0, 0, 2, 3, 9])
    r1 = p1 + p2
    r2 = p2 + p1
    r3 = x + y
    @test r1.coeffs == r2.coeffs == (r3.coeffs[end-2:end]...,)
    q1 = p1 * p2
    q2 = p2 * p1
    q3 = x * y
    @test q1.coeffs == q2.coeffs == (q3.coeffs[end-2:end]...,)
    r1 = p1 + p1
    r3 = x + x
    @test r1.coeffs == (r3.coeffs[end-2:end]...,)
    r1 = p1 * p1
    r3 = x * x
    @test r1.coeffs == (r3.coeffs[end-2:end]...,)
end

@testset "arithematics" begin
    for (a, b, c) in [
                    (TropicalF64(2), TropicalF64(8), TropicalF64(9)),
                    (CountingTropicalF64(2, 8), CountingTropicalF64(8, 9), CountingTropicalF64(9, 2)),
                    (Mod{17}(2), Mod{17}(8), Mod{17}(9)),
                    (Polynomial([0,1,2,3.0]), Polynomial([3,2.0]), Polynomial([1,7.0])),
                    (Max2Poly(1,2,3.0), Max2Poly(3,2,2.0), Max2Poly(4,7,1.0)),
                    (TruncatedPoly((1,2,3),3.0), TruncatedPoly((7,3,2),2.0), TruncatedPoly((1,4,7),1.0)),
                    (TropicalF64(5), TropicalF64(3), TropicalF64(-9)),
                    (CountingTropicalF64(5, 3), CountingTropicalF64(3, 9), CountingTropicalF64(-3, 2)),
                    (CountingTropical(5.0, ConfigSampler(StaticBitVector(rand(Bool, 10)))), CountingTropical(3.0, ConfigSampler(StaticBitVector(rand(Bool, 10)))), CountingTropical(-3.0, ConfigSampler(StaticBitVector(rand(Bool, 10))))),
                    (CountingTropical(5.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:3])), CountingTropical(3.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:4])), CountingTropical(-3.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:5]))),
                    ]
        @test is_commutative_semiring(a, b, c)
    end
end