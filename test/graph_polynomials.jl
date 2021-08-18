using GraphTensorNetworks, Test, OMEinsum, OMEinsumContractionOrders
using Mods, Polynomials, TropicalNumbers
using LightGraphs, Random

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

@testset "arithematics" begin
    for (a, b, c) in [
                    (TropicalF64(2), TropicalF64(8), TropicalF64(9)),
                    (CountingTropicalF64(2, 8), CountingTropicalF64(8, 9), CountingTropicalF64(9, 2)),
                    (Mod{17}(2), Mod{17}(8), Mod{17}(9)),
                    (Polynomial([0,1,2,3.0]), Polynomial([3,2.0]), Polynomial([1,7.0])),
                    (Max2Poly(1,2,3.0), Max2Poly(3,2,2.0), Max2Poly(4,7,1.0)),
                    (TropicalF64(5), TropicalF64(3), TropicalF64(-9)),
                    (CountingTropicalF64(5, 3), CountingTropicalF64(3, 9), CountingTropicalF64(-3, 2)),
                    (ConfigTropical{Float64,10,1}(5.0, BitVector(rand(Bool, 10))), ConfigTropical{Float64,10,1}(3.0, BitVector(rand(Bool, 10))), ConfigTropical{Float64,10,1}(-3.0, BitVector(rand(Bool, 10)))),
                    (CountingTropical(5.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:3])), CountingTropical(3.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:4])), CountingTropical(-3.0, ConfigEnumerator([StaticBitVector(rand(Bool, 10)) for j=1:5]))),
                    ]
        @test is_commutative_semiring(a, b, c)
    end
end

@testset "counting maximal IS" begin
    g = random_regular_graph(20, 3)
    cs = graph_polynomial(MaximalIndependence, Val(:fft), g; r=1.0, optmethod=:greedy)[]
    cs2 = graph_polynomial(MaximalIndependence, Val(:polynomial), g; optmethod=:greedy)[]
    cs3 = graph_polynomial(MaximalIndependence, Val(:finitefield), g; optmethod=:greedy)[]
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
    @test graph_polynomial(Matching, Val(:polynomial), g)[] == Polynomial([1,7,13,5])
end