using Test, GraphTensorNetworks
using GraphTensorNetworks: statictrues, staticfalses, StaticBitVector, onehotv

@testset "static bit vector" begin
    @test statictrues(StaticBitVector{3,1}) == trues(3)
    @test staticfalses(StaticBitVector{3,1}) == falses(3)
    x = rand(Bool, 131)
    y = rand(Bool, 131)
    a = StaticBitVector(x)
    b = StaticBitVector(y)
    a2 = BitVector(x)
    b2 = BitVector(y)
    for op in [|, &, ⊻]
        @test op(a, b) == op.(a2, b2)
    end
    @test onehotv(StaticBitVector{133,3}, 5) == (x = falses(133); x[5]=true; x)
end

