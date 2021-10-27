using GraphTensorNetworks, Test, OMEinsum, OMEinsumContractionOrders
using Mods, Polynomials, TropicalNumbers
using Graphs, Random
using GraphTensorNetworks: StaticBitVector

@testset "bond and vertex tensor" begin
    @test GraphTensorNetworks.misb(TropicalF64) == [TropicalF64(0) TropicalF64(0); TropicalF64(0) TropicalF64(-Inf)]
    @test GraphTensorNetworks.misv(TropicalF64(2.0)) == [TropicalF64(0), TropicalF64(2.0)]
end

@testset "graph generator" begin
    g = diagonal_coupled_graph(trues(3, 3))
    @test ne(g) == 20
    g = diagonal_coupled_graph((x = trues(3, 3); x[2,2]=0; x))
    @test ne(g) == 12
    @test length(GraphTensorNetworks.labels(Independence(g).code)) == 8
end

@testset "independence_polynomial" begin
    Random.seed!(2)
    g = random_regular_graph(10, 3)
    p1 = graph_polynomial(Independence(g), Val(:fitting))[]
    p2 = graph_polynomial(Independence(g), Val(:polynomial))[]
    p3 = graph_polynomial(Independence(g), Val(:fft))[]
    p4 = graph_polynomial(Independence(g), Val(:finitefield))[]
    @test p1 ≈ p2
    @test p1 ≈ p3
    @test p1 ≈ p4
end

@testset "counting maximal IS" begin
    g = random_regular_graph(20, 3)
    gp = MaximalIndependence(g, optimizer=KaHyParBipartite(sc_target=20))
    cs = graph_polynomial(gp, Val(:fft); r=1.0)[]
    gp = MaximalIndependence(g, optimizer=SABipartite(sc_target=20))
    cs2 = graph_polynomial(gp, Val(:polynomial))[]
    gp = MaximalIndependence(g, optimizer=GreedyMethod())
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

@testset "match polynomial" begin
    g = SimpleGraph(7)
    for (i,j) in [(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),(1,7)]
        add_edge!(g, i, j)
    end
    @test graph_polynomial(Matching(g), Val(:polynomial))[] == Polynomial([1,7,13,5])
    g = smallgraph(:petersen)
    @test graph_polynomial(Matching(g), Val(:polynomial))[].coeffs == [6, 90, 145, 75, 15, 1][end:-1:1]
end

@testset "spinglass" begin
    g = SimpleGraph(5)
    for (i,j) in [(1,2),(2,3),(3,4),(4,1),(1,5),(2,4)]
        add_edge!(g, i, j)
    end
    @test graph_polynomial(MaxCut(g), Val(:polynomial))[] == Polynomial([2,2,4,12,10,2])
    @test graph_polynomial(MaxCut(g), Val(:finitefield))[] == Polynomial([2,2,4,12,10,2])
end