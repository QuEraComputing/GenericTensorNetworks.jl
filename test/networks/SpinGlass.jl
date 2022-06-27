using GenericTensorNetworks, Test, Graphs

@testset "memory estimation" begin
    g = smallgraph(:petersen)
    J = rand(15)
    h = randn(10)
    gp = SpinGlass(g; h, J)
    M = zeros(10, 10)
    for (e,j) in zip(edges(g), J)
        M[e.src, e.dst] = j
    end; M += M'
    gp2 = SpinGlass(M, h)
    @test gp2.target.vertex_weights ≈ gp.target.vertex_weights
    @test gp2.target.edge_weights ≈ gp.target.edge_weights
    cfg(x) = [(x>>i & 1) for i=0:9]
    energies = [spinglass_energy(g, cfg(b); J=J, h) for b=0:1<<nv(g)-1]
    sorted_energies = sort(energies)
    @test solve(gp, SizeMax())[].n ≈ sorted_energies[end]
    @test solve(gp, SizeMin())[].n ≈ sorted_energies[1]
    @test getfield.(solve(gp, SizeMax(2))[].orders |> collect, :n) ≈ sorted_energies[end-1:end]
    @test getfield.(solve(gp, SingleConfigMax(2))[].orders |> collect, :n) ≈ sorted_energies[end-1:end]

    # integer weights
    J = NoWeight()
    h = ZeroWeight()
    gp = SpinGlass(g; h, J)
    energies = [spinglass_energy(g, cfg(b); J=J, h) for b=0:1<<nv(g)-1]
    sorted_energies = sort(energies)
    @test solve(gp, CountingMax(2))[].maxorder ≈ sorted_energies[end]
    @test solve(gp, CountingMin())[].n ≈ sorted_energies[1]
    @test solve(gp, ConfigsMax(2))[].maxorder ≈ sorted_energies[end]
    @test solve(gp, ConfigsMin())[].n ≈ sorted_energies[1]
    @test solve(gp, CountingAll())[] ≈ 1024
    poly = solve(gp, GraphPolynomial())[]
    @test poly.m[] == sorted_energies[1]
    @test poly.n[] == sorted_energies[end]
end