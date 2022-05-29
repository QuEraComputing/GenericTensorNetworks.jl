using GenericTensorNetworks, Test, Graphs
using GenericTensorNetworks: graph_polynomial

@testset "counting maximal IS" begin
    g = random_regular_graph(20, 3)
    gp = MaximalIS(g, optimizer=KaHyParBipartite(sc_target=20))
    cs = graph_polynomial(gp, Val(:fft); r=1.0)[]
    gp = MaximalIS(g, optimizer=SABipartite(sc_target=20))
    cs2 = graph_polynomial(gp, Val(:polynomial))[]
    gp = MaximalIS(g, optimizer=GreedyMethod())
    cs3 = graph_polynomial(gp, Val(:finitefield))[]
    cg = complement(g)
    cliques = maximal_cliques(cg)
    for i=1:20
        c = count(x->length(x)==i, cliques)
        if c > 0
            @test cs2.coeffs[i+1] == c
            @test cs3.coeffs[i+1] == c
            @test cs.coeffs[i+1] ≈ c
        end
    end
end

@testset "counting minimum maximal IS" begin
    g = smallgraph(:tutte)
    res = solve(MaximalIS(g), SizeMin())[]
    res2 = solve(MaximalIS(g), SizeMin(10))[]
    res3 = solve(MaximalIS(g), SingleConfigMin(10))[]
    poly = solve(MaximalIS(g), GraphPolynomial())[]
    @test poly == Polynomial([fill(0.0, 13)..., 2, 150, 7510, 71669, 66252, 14925, 571])
    @test res.n == 13
    @test res2.orders == Tropical.([13, 13, fill(14, 8)...])
    @test all(r->is_maximal_independent_set(g, r[2].c.data) && count_ones(r[2].c.data)==r[1].n, zip(res2.orders, res3.orders))
    @test solve(MaximalIS(g), CountingMin())[].c == 2
    min2 = solve(MaximalIS(g), CountingMin(3))[]
    @test min2.maxorder == 15
    @test min2.coeffs == (2, 150, 7510)

    for bounded in [false, true]
        println("bounded = ", bounded, ", configs max1")
        @test length(solve(MaximalIS(g), ConfigsMin(; bounded=bounded))[].c) == 2
        println("bounded = ", bounded, ", configs max3")
        cmin2 = solve(MaximalIS(g), ConfigsMin(3; bounded=bounded))[]
        @test cmin2.maxorder == 15
        @test length.(cmin2.coeffs) == (2, 150, 7510)

        println("bounded = ", bounded, ", single config min")
        c = solve(MaximalIS(g), SingleConfigMin(; bounded=bounded), T=Int64)[].c.data
        @test c ∈ cmin2.coeffs[1].data
        @test is_maximal_independent_set(g, c)
        @test count(!iszero, c) == 13
    end
end