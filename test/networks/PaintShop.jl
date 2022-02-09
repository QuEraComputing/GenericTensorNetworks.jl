using GraphTensorNetworks, Test

@testset "paint shop" begin
    labels = collect("abaccb")
    pb = PaintShop(labels)
    @test solve(pb, SizeMax())[] == Tropical(3.0)
    c = solve(pb, SingleConfigMax())[].c.data
    coloring = paint_shop_coloring_from_config(c)
    @test num_paint_shop_color_switch(labels, coloring) == 2
    @test StaticBitVector(Bool[0,1,1,0,1]) ∈ solve(pb, ConfigsMax())[].c.data
end