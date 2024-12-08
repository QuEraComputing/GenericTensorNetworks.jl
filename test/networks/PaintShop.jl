using GenericTensorNetworks, Test, ProblemReductions

@testset "paint shop" begin
    syms = collect("abaccb")
    pb = GenericTensorNetwork(PaintShop(syms))
    @test solve(pb, SizeMin())[] == Tropical(2.0)
    config = solve(pb, SingleConfigMin())[].c.data
    coloring = ProblemReductions.paint_shop_coloring_from_config(pb.problem, config)
    @test num_paint_shop_color_switch(syms, coloring) == 2
    @test bv"100" ∈ solve(pb, ConfigsMin())[].c.data
end