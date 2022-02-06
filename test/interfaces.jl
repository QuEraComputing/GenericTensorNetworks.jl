using GraphTensorNetworks
using Graphs, Test

@testset "independence problem" begin
    g = Graphs.smallgraph("petersen")
    for optimizer in (GreedyMethod(), TreeSA(ntrials=1))
        gp = Independence(g; optimizer=optimizer)
        res1 = solve(gp, SizeMax())[]
        res2 = solve(gp, CountingAll())[]
        res3 = solve(gp, CountingMax(1))[]
        res4 = solve(gp, CountingMax(2))[]
        res5 = solve(gp, GraphPolynomial(; method = :polynomial))[]
        res6 = solve(gp, SingleConfigMax())[]
        res7 = solve(gp, ConfigsMax(;bounded=false))[]
        res8 = solve(gp, ConfigsMax(2; bounded=false))[]
        res9 = solve(gp, ConfigsAll())[]
        res10 = solve(gp, GraphPolynomial(method=:fft))[]
        res11 = solve(gp, GraphPolynomial(method=:finitefield))[]
        res12 = solve(gp, SingleConfigMax(; bounded=true))[]
        res13 = solve(gp, ConfigsMax(;bounded=true))[]
        res14 = solve(gp, CountingMax(3))[]
        res15 = solve(gp, ConfigsMax(3))[]
        res16 = solve(gp, ConfigsMax(2; bounded=true))[]
        res17 = solve(gp, ConfigsMax(3; bounded=true))[]
        @test res1.n == 4
        @test res2 == 76
        @test res3.n == 4 && res3.c == 5
        @test res4.maxorder == 4 && res4.coeffs[1] == 30 && res4.coeffs[2]==5
        @test res5 == Polynomial([1.0, 10.0, 30, 30, 5])
        @test res6.c.data ∈ res7.c.data
        @test all(x->sum(x) == 4, res7.c.data)
        @test all(x->sum(x) == 3, res8.coeffs[1].data) && all(x->sum(x) == 4, res8.coeffs[2].data) && length(res8.coeffs[1].data) == 30 && length(res8.coeffs[2].data) == 5
        @test all(x->all(c->sum(c) == x[1]-1, x[2].data), enumerate(res9.coeffs))
        @test res10 ≈ res5
        @test res11 == res5
        @test res12.c.data ∈ res13.c.data
        @test res13.c == res7.c
        @test res14.maxorder == 4 && res14.coeffs[1]==30 && res14.coeffs[2] == 30 && res14.coeffs[3]==5
        @test all(x->sum(x) == 2, res15.coeffs[1].data) && all(x->sum(x) == 3, res15.coeffs[2].data) && all(x->sum(x) == 4, res15.coeffs[3].data) &&
                length(res15.coeffs[1].data) == 30 && length(res15.coeffs[2].data) == 30 && length(res15.coeffs[3].data) == 5
        @test all(x->sum(x) == 3, res16.coeffs[1].data) && all(x->sum(x) == 4, res16.coeffs[2].data) && length(res16.coeffs[1].data) == 30 && length(res16.coeffs[2].data) == 5
        @test all(x->sum(x) == 2, res17.coeffs[1].data) && all(x->sum(x) == 3, res17.coeffs[2].data) && all(x->sum(x) == 4, res17.coeffs[3].data) &&
                length(res17.coeffs[1].data) == 30 && length(res17.coeffs[2].data) == 30 && length(res17.coeffs[3].data) == 5
    end
end

@testset "save load" begin
    M = 10
    m = ConfigEnumerator([StaticBitVector(rand(Bool, 300)) for i=1:M])
    bm = GraphTensorNetworks.plain_matrix(m)
    rm = GraphTensorNetworks.raw_matrix(m)
    m1 = GraphTensorNetworks.from_raw_matrix(rm; bitlength=300, nflavors=2)
    m2 = GraphTensorNetworks.from_plain_matrix(bm; nflavors=2)
    @test m1 == m
    @test m2 == m
    save_configs("_test.bin", m; format=:binary)
    @test_throws ErrorException load_configs("_test.bin"; format=:binary)
    ma = load_configs("_test.bin"; format=:binary, bitlength=300, nflavors=2)
    @test ma == m

    save_configs("_test.txt", m; format=:text)
    mb = load_configs("_test.txt"; format=:text, nflavors=2)
    @test mb == m

    M = 10
    m = ConfigEnumerator([StaticElementVector(3, rand(1:3, 300)) for i=1:M])
    bm = GraphTensorNetworks.plain_matrix(m)
    rm = GraphTensorNetworks.raw_matrix(m)
    m1 = GraphTensorNetworks.from_raw_matrix(rm; bitlength=300, nflavors=3)
    m2 = GraphTensorNetworks.from_plain_matrix(bm; nflavors=3)
    @test m1 == m
    @test m2 == m
    @test Matrix(m) == bm
    @test Vector(m.data[1]) == bm[:,1]

    save_configs("_test.bin", m; format=:binary)
    @test_throws ErrorException load_configs("_test.bin"; format=:binary)
    ma = load_configs("_test.bin"; format=:binary, bitlength=300, nflavors=3)
    @test ma == m

    save_configs("_test.txt", m; format=:text)
    mb = load_configs("_test.txt"; format=:text, nflavors=3)
    @test mb == m
end

@testset "slicing" begin
    g = Graphs.smallgraph("petersen")
    gp = Independence(g; optimizer=TreeSA(nslices=5, ntrials=1))
    res1 = solve(gp, SizeMax())[]
    res2 = solve(gp, CountingAll())[]
    res3 = solve(gp, CountingMax(1))[]
    res4 = solve(gp, CountingMax(2))[]
    res5 = solve(gp, GraphPolynomial(; method = :polynomial))[]
    res6 = solve(gp, SingleConfigMax())[]
    res7 = solve(gp, ConfigsMax(;bounded=false))[]
    res8 = solve(gp, ConfigsMax(2; bounded=false))[]
    res9 = solve(gp, ConfigsAll())[]
    res10 = solve(gp, GraphPolynomial(method=:fft))[]
    res11 = solve(gp, GraphPolynomial(method=:finitefield))[]
    res12 = solve(gp, SingleConfigMax(; bounded=true))[]
    res13 = solve(gp, ConfigsMax(;bounded=true))[]
    res14 = solve(gp, CountingMax(3))[]
    res15 = solve(gp, ConfigsMax(3))[]
    res16 = solve(gp, ConfigsMax(2; bounded=true))[]
    res17 = solve(gp, ConfigsMax(3; bounded=true))[]
    @test res1.n == 4
    @test res2 == 76
    @test res3.n == 4 && res3.c == 5
    @test res4.maxorder == 4 && res4.coeffs[1] == 30 && res4.coeffs[2]==5
    @test res5 == Polynomial([1.0, 10.0, 30, 30, 5])
    @test res6.c.data ∈ res7.c.data
    @test all(x->sum(x) == 4, res7.c.data)
    @test all(x->sum(x) == 3, res8.coeffs[1].data) && all(x->sum(x) == 4, res8.coeffs[2].data) && length(res8.coeffs[1].data) == 30 && length(res8.coeffs[2].data) == 5
    @test all(x->all(c->sum(c) == x[1]-1, x[2].data), enumerate(res9.coeffs))
    @test res10 ≈ res5
    @test res11 == res5
    @test res12.c.data ∈ res13.c.data
    @test res13.c == res7.c
    @test res14.maxorder == 4 && res14.coeffs[1]==30 && res14.coeffs[2] == 30 && res14.coeffs[3]==5
    @test all(x->sum(x) == 2, res15.coeffs[1].data) && all(x->sum(x) == 3, res15.coeffs[2].data) && all(x->sum(x) == 4, res15.coeffs[3].data) &&
            length(res15.coeffs[1].data) == 30 && length(res15.coeffs[2].data) == 30 && length(res15.coeffs[3].data) == 5
    @test all(x->sum(x) == 3, res16.coeffs[1].data) && all(x->sum(x) == 4, res16.coeffs[2].data) && length(res16.coeffs[1].data) == 30 && length(res16.coeffs[2].data) == 5
    @test all(x->sum(x) == 2, res17.coeffs[1].data) && all(x->sum(x) == 3, res17.coeffs[2].data) && all(x->sum(x) == 4, res17.coeffs[3].data) &&
            length(res17.coeffs[1].data) == 30 && length(res17.coeffs[2].data) == 30 && length(res17.coeffs[3].data) == 5
end

