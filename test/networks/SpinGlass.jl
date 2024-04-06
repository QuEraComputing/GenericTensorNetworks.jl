using GenericTensorNetworks, Test, Graphs

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
    gp2 = spin_glass_from_matrix(M, h)
    @test gp2.target.vertex_weights ≈ gp.target.vertex_weights
    @test gp2.target.edge_weights ≈ gp.target.edge_weights
    cfg(x) = [(x>>i & 1) for i=0:9]
    energies = [spinglass_energy(g, cfg(b); J=J, h) for b=0:1<<nv(g)-1]
    sorted_energies = sort(energies)
    @test solve(gp, SizeMax())[].n ≈ sorted_energies[end]
    @test solve(gp, SizeMin())[].n ≈ sorted_energies[1]
    @test getfield.(solve(gp, SizeMax(2))[].orders |> collect, :n) ≈ sorted_energies[end-1:end]
    res = solve(gp, SingleConfigMax(2))[].orders |> collect
    @test getfield.(res, :n) ≈ sorted_energies[end-1:end]
    @test spinglass_energy(g, res[1].c.data; J, h) ≈ res[end-1].n
    @test spinglass_energy(g, res[2].c.data; J, h) ≈ res[end].n
    val, ind = findmax(energies)

    # integer weights
    J = UnitWeight()
    h = ZeroWeight()
    gp = GenericTensorNetwork(SpinGlass(g, J, h))
    energies = [spinglass_energy(g, cfg(b); J=J, h) for b=0:1<<nv(g)-1]
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
