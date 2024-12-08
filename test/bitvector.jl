using Test, GenericTensorNetworks
using GenericTensorNetworks: statictrues, staticfalses, StaticBitVector, onehotv, @bv_str, StaticElementVector

@testset "static bit vector" begin
    @test statictrues(StaticBitVector{3,1}) == trues(3)
    @test staticfalses(StaticBitVector{3,1}) == falses(3)
    @test_throws BoundsError statictrues(StaticBitVector{3,1})[4]
    #@test (@inbounds statictrues(StaticBitVector{3,1})[4]) == 0
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
    @test [StaticElementVector(3, [2,1,0,1])...] == [2,1,0,1]
    bl = rand(0:2,100)
    @test [StaticElementVector(3, bl)...] == bl
    bl = rand(1:15,100)
    xl = StaticElementVector(16, bl)
    @test typeof(xl) == StaticElementVector{100,4,7}
    @test [xl...] == bl

    @test bv"110_111" == StaticBitVector([1,1,0,1,1,1])
end

