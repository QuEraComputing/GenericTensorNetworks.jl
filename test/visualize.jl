using GenericTensorNetworks, Test, Graphs
using LuxorGraphPlot: Layout

@testset "visualize" begin
    graph = smallgraph(:petersen)
    @test show_graph(graph) isa Any
    configs = [rand(Bool, 10) for i=1:5, j=1:3]
    @test show_configs(graph, Layout(:stress), configs) isa Any
end

@testset "einsum" begin
    graph = smallgraph(:petersen)
    pb = GenericTensorNetwork(IndependentSet(graph))
    @test show_einsum(pb.code; optimal_distance=25, annotate_tensors=true) isa Any
    @test show_einsum(pb.code; tensor_locs=[(randn(), randn()) .* 40 for i=1:25]) isa Any
    @test show_einsum(pb.code; label_locs=[(randn(), randn()) .* 40 for i=1:10]) isa Any
    @test show_einsum(pb.code; tensor_locs=[(randn(), randn()) .*40 for i=1:25], label_locs=[(randn(), randn()) .* 40 for i=1:10]) isa Any
end