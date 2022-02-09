using GraphTensorNetworks, Test, OMEinsum, OMEinsumContractionOrders
using Mods, Polynomials, TropicalNumbers
using Graphs, Random
using GraphTensorNetworks: StaticBitVector

@testset "bond and vertex tensor" begin
    @test GraphTensorNetworks.misb(TropicalF64) == [TropicalF64(0) TropicalF64(0); TropicalF64(0) TropicalF64(-Inf)]
    @test GraphTensorNetworks.misv([one(TropicalF64), TropicalF64(2.0)]) == [TropicalF64(0), TropicalF64(2.0)]
end

@testset "graph generator" begin
    g = diagonal_coupled_graph(trues(3, 3))
    @test ne(g) == 20
    g = diagonal_coupled_graph((x = trues(3, 3); x[2,2]=0; x))
    @test ne(g) == 12
    @test length(GraphTensorNetworks.labels(IndependentSet(g).code)) == 8
end

@testset "independence_polynomial" begin
    Random.seed!(2)
    g = random_regular_graph(10, 3)
    p1 = graph_polynomial(IndependentSet(g), Val(:fitting))[]
    p2 = graph_polynomial(IndependentSet(g), Val(:polynomial))[]
    p3 = graph_polynomial(IndependentSet(g), Val(:fft))[]
    p4 = graph_polynomial(IndependentSet(g), Val(:finitefield))[]
    p5 = graph_polynomial(IndependentSet(g), Val(:finitefield); max_iter=1)[]
    @test p1 ≈ p2
    @test p1 ≈ p3
    @test p1 ≈ p4
    @test p1 ≈ p5

    # test overflow
    g = random_regular_graph(120, 3)
    gp = IndependentSet(g, optimizer=TreeSA(; ntrials=1); simplifier=MergeGreedy())
    p6 = graph_polynomial(gp, Val(:polynomial))[]
    p7 = graph_polynomial(gp, Val(:finitefield))[]
    @test p6 ≈ p7
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

@testset "paint shop" begin
    labels = collect("abaccb")
    pb = PaintShop(labels)
    @test solve(pb, SizeMax())[] == Tropical(3.0)
    @test StaticBitVector(Bool[0,1,1,0,1]) ∈ solve(pb, ConfigsMax())[].c.data
end