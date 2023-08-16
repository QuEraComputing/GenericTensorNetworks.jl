using GenericTensorNetworks, Test, Graphs

@testset "memory estimation" begin
    g = smallgraph(:petersen)
    ecliques = [[e.src, e.dst] for e in edges(g)]
    cliques = [ecliques..., [[v] for v in vertices(g)]...]
    J = rand(15)
    h = randn(10) .* 0.5
    weights = [-J..., h...]
    gp = HyperSpinGlass(10, cliques; weights)
    cfg(x) = [(x>>i & 1) for i=0:9]
    energies = [hyperspinglass_energy(cliques, cfg(b); weights) for b=0:1<<nv(g)-1]
    energies2 = [spinglass_energy(g, cfg(b); J, h) for b=0:1<<nv(g)-1]
    @test energies ≈ energies2
    sorted_energies = sort(energies)
    @test solve(gp, SizeMax())[].n ≈ sorted_energies[end]
    @test solve(gp, SizeMin())[].n ≈ sorted_energies[1]
    @test getfield.(solve(gp, SizeMax(2))[].orders |> collect, :n) ≈ sorted_energies[end-1:end]
    res = solve(gp, SingleConfigMax(2))[].orders |> collect
    @test getfield.(res, :n) ≈ sorted_energies[end-1:end]
    @test hyperspinglass_energy(cliques, res[1].c.data; weights) ≈ res[end-1].n
    @test hyperspinglass_energy(cliques, res[2].c.data; weights) ≈ res[end].n
    val, ind = findmax(energies)

    # integer weights
    weights = NoWeight()
    gp = HyperSpinGlass(10, ecliques; weights)
    energies = [hyperspinglass_energy(ecliques, cfg(b); weights) for b=0:1<<nv(g)-1]
    sorted_energies = sort(energies)
    @test solve(gp, CountingMax(2))[].maxorder ≈ sorted_energies[end]
    @test solve(gp, CountingMin())[].n ≈ sorted_energies[1]
    @test solve(gp, ConfigsMax(2))[].maxorder ≈ sorted_energies[end]
    @test solve(gp, ConfigsMin())[].n ≈ sorted_energies[1]
    @test solve(gp, CountingAll())[] ≈ 1024
    poly = solve(gp, GraphPolynomial(; method=:laurent))[]
    @test poly.order[] == sorted_energies[1]
    @test poly.order[] + length(poly.coeffs) == sorted_energies[end]
end

@testset "auto laurent" begin
    hyperDim = 2
    blockDim = 3
    graph = [[2, 12], [3, 11], [1, 5], [2, 4], [4, 5], [4, 8], [5, 7], [1, 9], [3, 7], [5, 9], [6, 8], [4, 9], [6, 7], [7, 12], [9, 10], [10, 12], [4, 6], [5, 6], [10, 11], [1, 2, 12], [1, 3, 11], [1, 11, 12], [2, 3, 10], [2, 10, 12], [3, 10, 11], [4, 8, 12], [4, 9, 11], [5, 7, 12], [7, 8, 12], [6, 7, 11], [7, 9, 11], [7, 11, 12], [5, 9, 10], [6, 8, 10], [8, 9, 10], [8, 10, 12], [9, 10, 11], [1, 2, 9], [1, 3, 8], [1, 8, 9], [2, 3, 7], [2, 7, 9], [3, 7, 8], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
    num_vertices = blockDim* 2^hyperDim
    weights = ones(Int, length(graph));
    problem = HyperSpinGlass(num_vertices, graph; weights);
    poly = solve(problem, GraphPolynomial())[]
    @test poly isa LaurentPolynomial

    weights = ones(length(graph));
    problem = HyperSpinGlass(num_vertices, graph; weights);
    poly = solve(problem, GraphPolynomial())[]
    @test poly isa LaurentPolynomial
end
