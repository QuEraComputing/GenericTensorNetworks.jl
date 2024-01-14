using GenericTensorNetworks, Test, OMEinsum
using GenericTensorNetworks.Mods, Polynomials, TropicalNumbers
using Graphs, Random
using GenericTensorNetworks: StaticBitVector
using LinearAlgebra

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
    Random.seed!(2)
    for (a, b, c) in [
                    (TropicalF64(2), TropicalF64(8), TropicalF64(9)),
                    (CountingTropicalF64(2, 8), CountingTropicalF64(8, 9), CountingTropicalF64(9, 2)),
                    (Mod{17}(2), Mod{17}(8), Mod{17}(9)),
                    (Polynomial([0,1,2,3.0]), Polynomial([3,2.0]), Polynomial([1,7.0])),
                    (Max2Poly(1,2,3.0), Max2Poly(3,2,2.0), Max2Poly(4,7,1.0)),
                    (TruncatedPoly((1,2,3),3.0), TruncatedPoly((7,3,2),2.0), TruncatedPoly((1,4,7),1.0)),
                    (TropicalF64(5), TropicalF64(3), TropicalF64(-9)),
                    (ExtendedTropical{2}(Tropical.([2.2,3.1])), ExtendedTropical{2}(Tropical.([-1.0, 4.0])), ExtendedTropical{2}(Tropical.([-Inf, 0.6]))),
                    (CountingTropicalF64(5, 3), CountingTropicalF64(3, 9), CountingTropicalF64(-3, 2)),
                    (CountingTropical(5.0, ConfigSampler(StaticBitVector(rand(Bool, 10)))), CountingTropical(3.0, ConfigSampler(StaticBitVector(rand(Bool, 10)))), CountingTropical(-3.0, ConfigSampler(StaticBitVector(rand(Bool, 10))))),
                    (CountingTropical(5.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:3])), CountingTropical(3.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:4])), CountingTropical(-3.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:5]))),
                    (ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:3]), ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:4]), ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:5])),
                    (SumProductTree(GenericTensorNetworks.SUM, [SumProductTree(StaticBitVector(rand(Bool, 10))) for j=1:2]...),
                        SumProductTree(GenericTensorNetworks.SUM, [SumProductTree(StaticBitVector(rand(Bool, 10))) for j=1:2]...),
                        SumProductTree(GenericTensorNetworks.SUM, [SumProductTree(StaticBitVector(rand(Bool, 10))) for j=1:2]...)
                        ),
                    (SumProductTree(GenericTensorNetworks.PROD, [SumProductTree(StaticBitVector(rand(Bool, 10))) for j=1:2]...),
                        SumProductTree(GenericTensorNetworks.PROD, [SumProductTree(StaticBitVector(rand(Bool, 10))) for j=1:2]...),
                        SumProductTree(GenericTensorNetworks.PROD, [SumProductTree(StaticBitVector(rand(Bool, 10))) for j=1:2]...)
                        ),
                    (SumProductTree(GenericTensorNetworks.PROD, [SumProductTree(OnehotVec{10, 2}(rand(1:10), 1)) for j=1:2]...),
                        SumProductTree(GenericTensorNetworks.PROD, [SumProductTree(OnehotVec{10, 2}(rand(1:10), 1)) for j=1:2]...),
                        SumProductTree(GenericTensorNetworks.PROD, [SumProductTree(OnehotVec{10, 2}(rand(1:10), 1)) for j=1:2]...)
                        ),
                    (SumProductTree(StaticBitVector(rand(Bool, 10))),
                        SumProductTree(StaticBitVector(rand(Bool, 10))),
                        SumProductTree(StaticBitVector(rand(Bool, 10))),
                        ),
                    ]
        @test is_commutative_semiring(a, b, c)
        @test false * a == zero(a)
        @test true * a == a
        @test a * false == zero(a)
        @test a * true == a
    end
    # the following tests are for Polynomial + ConfigEnumerator
    a = ConfigEnumerator([StaticBitVector(trues(10)) for j=1:3])
    @test 1 * a == a
    @test 0 * a == zero(a)
    @test a[1] == StaticBitVector(trues(10))
    @test copy(a) == a
    @test length.(a) == [10, 10, 10]
    @test map(x->length(x), a) == [10, 10, 10]

    # the following tests are for Polynomial + ConfigEnumerator
    a = SumProductTree(StaticBitVector(trues(10)))
    @test 1 * a == a
    @test 0 * a == zero(a)
    @test copy(a) == a
    @test length(a) == 1
    @test length(a + a) == 2
    @test length(a * a) == 1
    print((a+a) * a)
    b = a + a
    @test GenericTensorNetworks.num_nodes(b * b) == 3
 
    a = ConfigSampler(StaticBitVector(rand(Bool, 10)))
    @test 1 * a == a
    @test 0 * a == zero(a)

    println(Max2Poly{Float64,Float64}(1, 1, 1))
    @test abs(Mod{5}(2)) == Mod{5}(2)
    @test Mod{5}(12) < Mod{5}(8)
end

@testset "powers" begin
    x = ConfigEnumerator([bv"00111"])
    y = 2.0
    @test x ^ 0 == one(x)
    @test Base.literal_pow(^, x, Val(2.0)) == x
    @test x ^ y == x
    x = SumProductTree(bv"00111")
    @test x ^ 0 == one(x)
    @test Base.literal_pow(^, x, Val(2.0)) == x
    @test x ^ y == x
    x = ConfigSampler(bv"00111")
    @test x ^ 0 == one(x)
    @test Base.literal_pow(^, x, Val(2.0)) == x
    @test x ^ y == x

    x = ExtendedTropical{3}(Tropical.([1.0, 2.0, 3.0]))
    @test x ^ 1 == x
    @test x ^ 0 == one(x)
    @test x ^ 1.0 == x
    @test x ^ 0.0 == one(x)
    @test x ^ 2 == ExtendedTropical{3}(Tropical.([2.0, 4.0, 6.0]))
    @test Base.literal_pow(^, x, Val(2.0)) == ExtendedTropical{3}(Tropical.([2.0, 4.0, 6.0]))
    @test Base.literal_pow(^, x, Val(2.0)) == x ^ y
    
    # truncated poly
    x = TruncatedPoly((2,3), 5)
    @test x ^ 2 == TruncatedPoly((12,9), 10)
    @test_throws ErrorException x ^ -2
    x = TruncatedPoly((0.0,3.0), 5)
    @test x ^ 2 == TruncatedPoly((0,9), 10)
    @test x ^ -2 == TruncatedPoly((0.0,3^-2), -10)
end

@testset "push coverage" begin
    @test abs(Mod{5}(2)) == Mod{5}(2)
    @test one(ConfigSampler(bv"11100")) == ConfigSampler(bv"00000")
    T = SumProductTree{OnehotVec{5,2}}
    @test collect(one(T(GenericTensorNetworks.ZERO))) == collect(SumProductTree(bv"00000"))
    @test iszero(copy(T(GenericTensorNetworks.ZERO)))
    x = SumProductTree(bv"00111")
    @test copy(x) == x
    @test copy(x) !== x
    @test !iszero(x)
    y = x + x
    @test copy(y) == y
    @test copy(y) !== y
    @test !iszero(y)
    println((x * x) * zero(x))
end

@testset "Truncated Tropical" begin
    # +
    a = ExtendedTropical{3}(Tropical.([1,2,3]))
    b = ExtendedTropical{3}(Tropical.([4,5,6]))
    c = ExtendedTropical{3}(Tropical.([0,1,2]))
    @test a + b == ExtendedTropical{3}(Tropical.([4,5,6]))
    @test b + a == ExtendedTropical{3}(Tropical.([4,5,6]))
    @test c + a == ExtendedTropical{3}(Tropical.([2,2,3]))
    @test a + c == ExtendedTropical{3}(Tropical.([2,2,3]))

    # *
    function naive_mul(a, b)
        K = length(a)
        return sort!(vec([x*y for x in a, y in b]))[end-K+1:end]
    end
    d = ExtendedTropical{3}(Tropical.([0,1,20]))
    @test naive_mul(a.orders, b.orders) == (a * b).orders
    @test naive_mul(b.orders, a.orders) == (b * a).orders
    @test naive_mul(a.orders, d.orders) == (a * d).orders
    @test naive_mul(d.orders, a.orders) == (d * a).orders
    @test naive_mul(d.orders, d.orders) == (d * d).orders
    for i=1:20
        a = ExtendedTropical{100}(Tropical.(sort!(randn(100))))
        b = ExtendedTropical{100}(Tropical.(sort!(randn(100))))
        @test naive_mul(a.orders, b.orders) == (a * b).orders
    end
end

@testset "count geq" begin
    A = collect(1:10)
    B = collect(2:2:20)
    low = 20
    c, _ = GenericTensorNetworks.count_geq(A, B, 1, low, false)
    @test c == count(x->x>=low, vec([a*b for a in A, b in B]))
    res = similar(A, c)
    @test sort!(GenericTensorNetworks.collect_geq!(res, A, B, 1, low)) == sort!(filter(x->x>=low, vec([a*b for a in A, b in B])))
end


# check the correctness of sampling
@testset "generate samples" begin
    Random.seed!(2)
    g = smallgraph(:petersen)
    gp = IndependentSet(g)
    t = solve(gp, ConfigsAll(tree_storage=true))[]
    cs = solve(gp, ConfigsAll())[]
    @test length(t) == 76
    samples = generate_samples(t, 10000)
    counts = zeros(5)
    for sample in samples
        counts[count_ones(sample)+1] += 1
    end
    @test isapprox(counts, [1,10,30,30,5] .* 10000 ./ 76, rtol=0.05)
    hd1 = hamming_distribution(samples, samples) |> normalize
    hd2 = hamming_distribution(cs, cs) |> normalize
    @test isapprox(hd1, hd2, atol=0.01)
end

@testset "LaurentPolynomial" begin
    @test GenericTensorNetworks._x(LaurentPolynomial{Float64, :x}; invert=false) == GenericTensorNetworks._x(Polynomial{Float64, :x}; invert=false)
end