using GenericTensorNetworks, Test, OMEinsum
using GenericTensorNetworks.Mods, Polynomials, TropicalNumbers
using Graphs, Random
using GenericTensorNetworks: StaticBitVector, graph_polynomial

@testset "graph generator" begin
    g = diagonal_coupled_graph(trues(3, 3))
    @test ne(g) == 20
    g = diagonal_coupled_graph((x = trues(3, 3); x[2,2]=0; x))
    @test ne(g) == 12
    @test length(uniquelabels(GenericTensorNetwork(IndependentSet(g)).code)) == 8
end

@testset "independence_polynomial" begin
    Random.seed!(2)
    g = random_regular_graph(10, 3)
    tn = GenericTensorNetwork(IndependentSet(g))
    p1 = graph_polynomial(tn, Val(:fitting))[]
    p2 = graph_polynomial(tn, Val(:polynomial))[]
    p3 = graph_polynomial(tn, Val(:fft))[]
    p4 = graph_polynomial(tn, Val(:finitefield))[]
    p5 = graph_polynomial(tn, Val(:finitefield); max_iter=1)[]
    p8 = solve(tn, GraphPolynomial(; method=:laurent))[]
    tn9 = GenericTensorNetwork(IndependentSet(g, fill(-1, 10)))
    p9 = solve(tn9, GraphPolynomial(; method=:polynomial))[]
    tn10 = GenericTensorNetwork(IndependentSet(g, rand([1,-1], 10)))
    p10 = solve(tn10, GraphPolynomial(; method=:laurent))[]
    @test p1 ≈ p2
    @test p1 ≈ p3
    @test p1 ≈ p4
    @test p1 ≈ p5
    @test p1 ≈ p8
    @test p1 ≈ GenericTensorNetworks.invert_polynomial(p9)
    @test sum(p10.coeffs) == sum(p1.coeffs)

    # test overflow
    g = random_regular_graph(120, 3)
    gp = GenericTensorNetwork(IndependentSet(g), optimizer=TreeSA(; ntrials=1))
    p6 = graph_polynomial(gp, Val(:polynomial))[]
    p7 = graph_polynomial(gp, Val(:finitefield))[]
    @test p6.coeffs ≈ p7.coeffs
end

@testset "open indices" begin
    g = SimpleGraph(3)
    for (i,j) in [(1,2), (2,3)]
        add_edge!(g, i, j)
    end
    tn = GenericTensorNetwork(IndependentSet(g); openvertices=(1,3))
    m1 = solve(tn, GraphPolynomial())
    m2 = solve(tn, GraphPolynomial(;method=:polynomial))
    @test m1 == m2
end