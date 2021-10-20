using GraphTensorNetworks
using LightGraphs, Test

@testset "independence problem" begin
    g = LightGraphs.smallgraph("petersen")
    gp = Independence(g; optimizer=GreedyMethod())
    res1 = solve(gp, "size max")[]
    res2 = solve(gp, "counting sum")[]
    res3 = solve(gp, "counting max")[]
    res4 = solve(gp, "counting max2")[]
    res5 = solve(gp, "counting all")[]
    res6 = solve(gp, "config max")[]
    res7 = solve(gp, "configs max")[]
    res8 = solve(gp, "configs max2")[]
    res9 = solve(gp, "configs all")[]
    res10 = solve(gp, "counting all (fft)")[]
    res11 = solve(gp, "counting all (finitefield)")[]
    res12 = solve(gp, "config max (bounded)")[]
    res13 = solve(gp, "configs max (bounded)")[]
    @test res1.n == 4
    @test res2 == 76
    @test res3.n == 4 && res3.c == 5
    @test res4.maxorder == 4 && res4.a == 30 && res4.b==5
    @test res5 == Polynomial([1.0, 10.0, 30, 30, 5])
    @test res6.c.data ∈ res7.c.data
    @test all(x->sum(x) == 4, res7.c.data)
    @test all(x->sum(x) == 3, res8.a.data) && all(x->sum(x) == 4, res8.b.data) && length(res8.a.data) == 30 && length(res8.b.data) == 5
    @test all(x->all(c->sum(c) == x[1]-1, x[2].data), enumerate(res9.coeffs))
    @test res10 ≈ res5
    @test res11 == res5
    @test res12.c.data ∈ res13.c.data
    @test res13.c.data == res7.c.data
end

@testset "save load" begin
    M = 10
    m = ConfigEnumerator([StaticBitVector(rand(Bool, 300)) for i=1:M])
    bm = GraphTensorNetworks.plain_matrix(m)
    rm = GraphTensorNetworks.raw_matrix(m)
    m1 = GraphTensorNetworks.from_raw_matrix(rm; len=300, nflavors=2)
    m2 = GraphTensorNetworks.from_plain_matrix(bm; nflavors=2)
    @test m1 == m
    @test m2 == m
    save_configs("_test.bin", m; format=:binary)
    @test_throws ErrorException load_configs("_test.bin"; format=:binary)
    ma = load_configs("_test.bin"; format=:binary, len=300, nflavors=2)
    @test ma == m

    save_configs("_test.txt", m; format=:text)
    mb = load_configs("_test.txt"; format=:text, nflavors=2)
    @test mb == m

    M = 10
    m = ConfigEnumerator([StaticElementVector(3, rand(1:3, 300)) for i=1:M])
    bm = GraphTensorNetworks.plain_matrix(m)
    rm = GraphTensorNetworks.raw_matrix(m)
    m1 = GraphTensorNetworks.from_raw_matrix(rm; len=300, nflavors=3)
    m2 = GraphTensorNetworks.from_plain_matrix(bm; nflavors=3)
    @test m1 == m
    @test m2 == m
    @test Matrix(m) == bm
    @test Vector(m.data[1]) == bm[:,1]

    save_configs("_test.bin", m; format=:binary)
    @test_throws ErrorException load_configs("_test.bin"; format=:binary)
    ma = load_configs("_test.bin"; format=:binary, len=300, nflavors=3)
    @test ma == m

    save_configs("_test.txt", m; format=:text)
    mb = load_configs("_test.txt"; format=:text, nflavors=3)
    @test mb == m
end