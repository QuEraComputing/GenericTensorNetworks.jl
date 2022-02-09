using GraphTensorNetworks, Test, Graphs

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

