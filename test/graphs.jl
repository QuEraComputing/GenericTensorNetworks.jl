using GraphTensorNetworks, Test, Graphs

@testset "special graphs" begin
    g = random_square_lattice_graph(10, 10, 0.5)
    @test nv(g) == 50
    g = random_diagonal_coupled_graph(10, 10, 0.5)
    @test nv(g) == 50
    g = unit_disk_graph([(0.1, 0.2), (0.2, 0.3), (1.2, 1.4)], 1.0)
    @test ne(g) == 1
    @test nv(g) == 3
end