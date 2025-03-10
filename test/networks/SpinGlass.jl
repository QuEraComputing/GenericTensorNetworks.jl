using GenericTensorNetworks, Test, Graphs

@testset "memory estimation" begin
    g = smallgraph(:petersen)
    J = rand(15)
    h = randn(10) .* 0.5
    problem = SpinGlass(g, J, h)
    gp = GenericTensorNetwork(problem)
    cfg(x) = [x>>i & 1 for i=0:9]
    energies = [energy(problem, cfg(b)) for b=0:1<<nv(g)-1]
    sorted_energies = sort(energies)
    @test solve(gp, SizeMax())[].n ≈ sorted_energies[end]
    @test solve(gp, SizeMin())[].n ≈ sorted_energies[1]
    @test getfield.(solve(gp, SizeMax(2))[].orders |> collect, :n) ≈ sorted_energies[end-1:end]
    res = solve(gp, SingleConfigMax(2))[].orders |> collect
    @test getfield.(res, :n) ≈ sorted_energies[end-1:end]
    cfg2(x) = Int.(x)
    @test energy(problem, cfg2(res[1].c.data)) ≈ res[end-1].n
    @test energy(problem, cfg2(res[2].c.data)) ≈ res[end].n
    val, ind = findmax(energies)

    J = UnitWeight(15)
    h = zeros(Int, 10)
    problem = SpinGlass(g, J, h)
    gp = GenericTensorNetwork(problem)
    energies = [energy(problem, cfg(b)) for b=0:1<<nv(g)-1]
    sorted_energies = sort(energies)
    @test solve(gp, CountingMax(2))[].maxorder ≈ sorted_energies[end]
    @test solve(gp, CountingMin())[].n ≈ sorted_energies[1]
    @test solve(gp, ConfigsMax(2))[].maxorder ≈ sorted_energies[end]
    @test solve(gp, ConfigsMin())[].n ≈ sorted_energies[1]
    @test solve(gp, CountingAll())[] ≈ 1024
    poly = solve(gp, GraphPolynomial(; method=:laurent))[]
    @test poly.order[] == sorted_energies[1]
    @test poly.order[] + length(poly.coeffs) - 1 == sorted_energies[end]
end

@testset "auto laurent" begin
    hyperDim = 2
    blockDim = 3
    edgs = [[2, 12], [3, 11], [1, 5], [2, 4], [4, 5], [4, 8], [5, 7], [1, 9], [3, 7], [5, 9], [6, 8], [4, 9], [6, 7], [7, 12], [9, 10], [10, 12], [4, 6], [5, 6], [10, 11], [1, 2, 12], [1, 3, 11], [1, 11, 12], [2, 3, 10], [2, 10, 12], [3, 10, 11], [4, 8, 12], [4, 9, 11], [5, 7, 12], [7, 8, 12], [6, 7, 11], [7, 9, 11], [7, 11, 12], [5, 9, 10], [6, 8, 10], [8, 9, 10], [8, 10, 12], [9, 10, 11], [1, 2, 9], [1, 3, 8], [1, 8, 9], [2, 3, 7], [2, 7, 9], [3, 7, 8], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
    graph = HyperGraph(maximum(maximum.(edgs)), edgs)
    num_vertices = blockDim* 2^hyperDim
    J = ones(Int, ne(graph));
    problem = GenericTensorNetwork(SpinGlass(graph, J, zeros(Int, nv(graph))));
    poly = solve(problem, GraphPolynomial())[]
    @test poly isa LaurentPolynomial

    J = ones(Int, ne(graph));
    problem = GenericTensorNetwork(SpinGlass(graph, J, zeros(Int, nv(graph))))
    poly = solve(problem, GraphPolynomial())[]
    @test poly isa LaurentPolynomial
end

@testset "memory estimation" begin
    g = smallgraph(:petersen)
    J = rand(15)
    h = randn(10) .* 0.5
    gp = GenericTensorNetwork(SpinGlass(g, J, h))
    @test contraction_complexity(gp).sc <= 5
    M = zeros(10, 10)
    for (e,j) in zip(edges(g), J)
        M[e.src, e.dst] = j
    end; M += M'
    gp2 = GenericTensorNetwork(spin_glass_from_matrix(M, h))
    @test gp2.problem.J ≈ gp.problem.J
    cfg(x) = [(x>>i & 1) for i=0:9]
    energies = [energy(gp.problem, cfg(b)) for b=0:1<<nv(g)-1]
    sorted_energies = sort(energies)
    @test solve(gp, SizeMax())[].n ≈ sorted_energies[end]
    @test solve(gp, SizeMin())[].n ≈ sorted_energies[1]
    @test getfield.(solve(gp, SizeMax(2))[].orders |> collect, :n) ≈ sorted_energies[end-1:end]
    res = solve(gp, SingleConfigMax(2))[].orders |> collect
    @test getfield.(res, :n) ≈ sorted_energies[end-1:end]
    cfg2(x) = Int.(x)
    @test energy(gp.problem, cfg2(res[1].c.data)) ≈ res[end-1].n
    @test energy(gp.problem, cfg2(res[2].c.data)) ≈ res[end].n
    val, ind = findmax(energies)

    J = UnitWeight(ne(g))
    h = zeros(Int, nv(g))
    gp = GenericTensorNetwork(SpinGlass(g, J, h))
    energies = [energy(gp.problem, cfg(b)) for b=0:1<<nv(g)-1]
    sorted_energies = sort(energies)
    @test solve(gp, CountingMax(2))[].maxorder ≈ sorted_energies[end]
    @test solve(gp, CountingMin())[].n ≈ sorted_energies[1]
    @test solve(gp, ConfigsMax(2))[].maxorder ≈ sorted_energies[end]
    @test solve(gp, ConfigsMin())[].n ≈ sorted_energies[1]
    @test solve(gp, CountingAll())[] ≈ 1024
    poly = solve(gp, GraphPolynomial())[]
    @test poly.order[] == sorted_energies[1]
    @test poly.order[] + length(poly.coeffs) - 1 == sorted_energies[end]
end
