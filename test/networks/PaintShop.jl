using GenericTensorNetworks, Test

@testset "paint shop" begin
    syms = collect("abaccb")
    pb = PaintShop(syms)
    @test get_weights(pb) == UnitWeight()
    @test get_weights(chweights(pb, fill(3, 15))) == UnitWeight()
    @test solve(pb, SizeMin())[] == Tropical(2.0)
    config = solve(pb, SingleConfigMin())[].c.data
    coloring = paint_shop_coloring_from_config(pb, config)
    @test num_paint_shop_color_switch(syms, coloring) == 2
    @test bv"100" ∈ solve(pb, ConfigsMin())[].c.data
end