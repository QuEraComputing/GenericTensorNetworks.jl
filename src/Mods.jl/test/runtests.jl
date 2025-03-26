using Test
using GenericTensorNetworks.Mods

@testset "Constructors" begin
    @test one(Mod{17}) == Mod{17}(1)
    @test one(Mod{17}) != Mod{17}(2)
    @test one(Mod{17}) == Mod{17}(18)
    @test oneunit(Mod{17}) == Mod{17}(1)
    @test zero(Mod{17}) == 0
    @test iszero(zero(Mod{17}))
    @test Mod{17}(1) == Mod{17}(1, 0)
    @test Mod{17}(1, 2) == 1 + 2im
    a = Mod{17}(3)
    @test typeof(a) == Mod{17,Int}
    a = Mod{17}(3, 2)
    @test typeof(a) == GaussMod{17,Int}
    a = zero(Mod{17})
    @test typeof(a) == Mod{17,Int}
    a = Mod{17}(1 // 2 + (3 // 4)im)
    @test typeof(a) == GaussMod{17,Int}
end


@testset "Mod arithmetic" begin
    p = 23
    a = Mod{p}(2)
    b = Mod{p}(25)
    @test a == b
    @test a == 2
    @test a == -21

    b = Mod{p}(20)
    @test a + b == 22
    @test a - b == -18
    @test a + a == 2a
    @test 0 - a == -a

    @test a * b == Mod{p}(17)
    @test (a / b) * b == a
    @test (b // a) * (2 // 1) == b
    @test a * (2 // 3) == (2a) * inv(Mod{p}(3))

    @test is_invertible(a)
    @test !is_invertible(Mod{10}(4))

    @test a^(p - 1) == 1
    @test a^(-1) == inv(a)

    @test Mod{p}(3 // 7) == 3 // 7
    @test isequal(3 // 7, Mod{p}(3 // 7))

    @test -Mod{13,Int}(typemin(Int)) == -Mod{13}(mod(typemin(Int), 13))
end

@testset "Mod arithmetic with UInt" begin
    p = UInt(23)
    a = Mod{p,UInt}(2)
    b = Mod{p,UInt}(25)
    @test a == b
    @test a == 2
    @test a == 25

    b = Mod{p,UInt}(20)
    @test a + b == 22
    @test a - b == 28
    @test a + a == 2a
    @test 0 - a == -a

    @test a * b == Mod{p,UInt}(17)
    @test (a / b) * b == a
    @test (b // a) * (2 // 1) == b
    @test a * (2 // 3) == (2a) * inv(Mod{p,UInt}(3))

    @test is_invertible(a)
    @test !is_invertible(Mod{10}(4))

    @test a^(p - 1) == 1
    @test a^(-1) == inv(a)
end

@testset "GaussMod arithmetic" begin
    @test one(GaussMod{6}) == Mod{6}(1 + 0im)
    @test zero(GaussMod{6}) == Mod{6}(0 + 0im)
    @test GaussMod{6}(2 + 3im) == Mod{6}(2 + 3im)
    @test GaussMod{6,Int}(2 + 3im) == Mod{6}(2 + 3im)
    @test rand(GaussMod{6}) isa GaussMod
    @test eltype(rand(GaussMod{6}, 4, 4)) <: GaussMod
    p = 23
    a = GaussMod{p}(3 - im)
    b = Mod{p}(5 + 5im)

    @test a + b == 8 + 4im
    @test a + Mod{p}(11) == Mod{p}(14, 22)
    @test -a == 20 + im
    @test a - b == Mod{p}(3 - im - 5 - 5im)

    @test a * b == Mod{p}((3 - im) * (5 + 5im))
    @test a / b == Mod{p}((3 - im) // (5 + 5im))

    @test a^(p * p - 1) == 1
    @test is_invertible(a)
    @test a * inv(a) == 1

    @test a / (1 + im) == a / Mod{p}(1 + im)
    @test imag(a * a') == 0
end

@testset "Large Modulus" begin

    p = 9223372036854775783   # This is a large prime
    x = Mod{p}(-2)
    @test x * x == 4
    @test x + x == -4
    @test x / x == 1
    @test x / 3 == x / Mod{p}(3)
    @test (x / 3) * (3 // x) == 1
    @test x // x == value(x) / x
    @test x^4 == 16
    @test x^(p - 1) == 1   # Fermat Little Theorem test
    @test 2x == x + x
    @test x - x == 0
    y = inv(x)
    @test x * y == 1
    @test x + p == x
    @test x * p == 0
    @test p - x == x - 2x

    p = 9223372036854775783   # This is a large prime
    x = Mod{p}(-2 + 0im)
    @test x * x == 4
    @test x + x == -4
    @test x / x == 1
    @test x / 3 == x / Mod{p}(3)
    @test (x / 3) * (3 // x) == 1
    @test x // x == value(x) / x
    @test x^4 == 16
    @test x^(p - 1) == 1   # Fermat Little Theorem test
    @test 2x == x + x
    @test x - x == 0
    y = inv(x)
    @test x * y == 1
    @test x + p == x
    @test x * p == 0
    @test p - x == x - 2x

    @test 0 <= value(rand(Mod{p})) < p

end



@testset "CRT" begin
    p = 23
    q = 91
    a = Mod{p}(17)
    b = Mod{q}(32)

    @test CRT(Int32, [], []) === Int32(0)
    x = CRT(a, b)
    @test typeof(x) <: BigInt
    @test a == mod(x, p)
    @test b == mod(x, q)

    c = Mod{101}(86)
    x = CRT(Int, a, b, c)
    @test typeof(x) <: Int

    @test a == mod(x, p)
    @test b == mod(x, q)
    @test c == mod(x, 101)

    @test CRT(
        BigInt,
        Mod{9223372036854775783}(9223372036854775782),
        Mod{9223372036854775643}(9223372036854775642),
    ) == 85070591730234614113402964855534653468
end


@testset "Matrices" begin
    M = zeros(Mod{11}, 3, 3)
    @test sum(M) == 0

    M = ones(Mod{11}, 5, 5)
    @test sum(M) == 3

    M = rand(Mod{11}, 5, 6)
    @test size(M) == (5, 6)

    A = rand(Mod{17}, 5, 5)
    X = values.(A)
    @test sum(X) == sum(Mod{17}.(A))
end

@testset "Hashing/Iterating" begin
    x = Mod{17}(11)
    y = x + 0im
    @test x == y
    @test hash(x) == hash(y)
    @test typeof(x) !== typeof(y)

    A = Set([x, y])
    @test length(A) == 1

    v = [Mod{10}(t) for t = 1:15]
    w = [Mod{10}(t + 0im) for t = 1:15]
    S = Set(v)
    T = Set(w)
    @test length(S) == 10
    @test S == T
    @test union(S, T) == intersect(S, T)
end


@testset "Iteration" begin
    @test sum(k for k in Mod{7}) == zero(Mod{7})
    @test collect(Mod{7}) == Mod{7}.(0:6)
    @test prod(k for k in Mod{7} if k != 0) == -1  # Wilson's theorem

    @test sum(GaussMod{7}) == 0
    @test length(GaussMod{5}) == 25
end

@testset "Linear system solution" begin
    A = Mod{13}.([1 2; 3 -4])
    b = Mod{13}.([3, 4])
    x = A \ b
    @test A * x == b
    @test inv(A) * b == x
end

@testset "isapprox" begin
    @test Mod{7}(3) ≈ Mod{7}(3)
    @test Mod{7}(3) ≈ Mod{7}(10)
    @test Mod{7}(3) ≈ Mod{7}(10) atol=1
    @test Mod{7}(3) ≈ Mod{7}(11) atol=1
    @test !isapprox(Mod{7}(3), Mod{7}(11), atol=0)
    @test !(Mod{7}(3) ≈ Mod{7}(11))
    @test Mod{7}(3) ≈ Mod{7}(11) atol=2
end
