using GraphTensorNetworks, Test, Graphs

@testset "visualize" begin
    locations = [(1.0, 2.0), (2.0, 3.0)]
    @test show_graph(locations, [(1, 2)]) isa Any
    @test show_graph(smallgraph(:petersen)) isa Any
    @test show_gallery(graph, (2,4); locs=locations, vertex_configs=[rand(Bool, 15) for i=1:10], edge_configs=[rand(Bool, 15) for i=1:10]) isa Any
end