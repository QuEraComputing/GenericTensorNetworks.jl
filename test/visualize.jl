using GenericTensorNetworks, Test, Graphs

@testset "visualize" begin
    locations = [(1.0, 2.0), (2.0, 3.0)]
    @test show_graph(locations, [(1, 2)]) isa Any
    @test show_graph([], []) isa Any
    @test show_graph([], []; format=:pdf) isa Any
    @test show_graph([], []; filename=tempname()*".svg") isa Any
    graph = smallgraph(:petersen)
    @test show_graph(graph) isa Any
    @test show_gallery(graph, (2,4); vertex_configs=[rand(Bool, 15) for i=1:10], edge_configs=[rand(Bool, 15) for i=1:10]) isa Any
end

@testset "einsum" begin
    graph = smallgraph(:petersen)
    pb = IndependentSet(graph)
    @test show_einsum(pb.code; optimal_distance=2, annotate_tensors=true) isa Any
    @test show_einsum(pb.code; tensor_locs=[(randn(), randn()) .* 2 for i=1:25]) isa Any
    @test show_einsum(pb.code; label_locs=[(randn(), randn()) .* 2 for i=1:10]) isa Any
    @test show_einsum(pb.code; tensor_locs=[(randn(), randn()) for i=1:25], label_locs=[(randn(), randn()) for i=1:10]) isa Any
end