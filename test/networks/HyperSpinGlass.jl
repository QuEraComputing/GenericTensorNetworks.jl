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
    @test poly.m[] == sorted_energies[1]
    @test poly.n[] == sorted_energies[end]
end