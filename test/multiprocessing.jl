using GraphTensorNetworks.SimpleMultiprocessing, Test

@testset "multiprocessing" begin
    results = multiprocess_run(x->x^2, collect(1:5))
    @test results == [1, 2, 3, 4, 5] .^ 2
end