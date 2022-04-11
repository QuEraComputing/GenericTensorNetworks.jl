using Test, OMEinsum, GenericTensorNetworks, TropicalNumbers, Random
using GenericTensorNetworks: cached_einsum, generate_masktree, masked_einsum, CacheTree, uniformsize, bounding_contract

@testset "cached einsum" begin
    xs = map(x->Tropical.(x), [randn(2,2), randn(2), randn(2,2), randn(2,2), randn(2,2)])
    code = ein"((ij,j),jk, kl), ii->kli"
    size_dict = uniformsize(code, 2)
    c = cached_einsum(code, xs, size_dict)
    @test c.content == code(xs...)
    mt = generate_masktree(AllConfigs{1}(), code, c, rand(Bool,2,2,2), size_dict)
    @test mt isa CacheTree{Bool}
    y = masked_einsum(code, xs, mt, size_dict)
    @test y isa AbstractArray
end

@testset "bounding contract" begin
    for seed in 1:100
        Random.seed!(seed)
        xs = map(x->TropicalF64.(x), [rand(1:5,2,2), rand(1:5,2), rand(1:5,2,2), rand(1:5,2,2), rand(1:5,2,2)])
        code = ein"((ij,j),jk, kl), ii->kli"
        y1 = code(xs...)
        y2 = bounding_contract(AllConfigs{1}(), code, xs, BitArray(ones(Bool,2,2,2)), xs)
        @test y1 ≈ y2
    end
    rawcode = IndependentSet(random_regular_graph(10, 3); optimizer=nothing).code
    optcode = IndependentSet(random_regular_graph(10, 3); optimizer=GreedyMethod()).code
    xs = map(OMEinsum.getixs(rawcode)) do ix
        length(ix)==1 ? GenericTensorNetworks.misv([one(TropicalF64), TropicalF64(1.0)]) : GenericTensorNetworks.misb(TropicalF64)
    end
    y1 = rawcode(xs...)
    y2 = bounding_contract(AllConfigs{1}(), rawcode, xs, BitArray(fill(true)), xs)
    @test y1 ≈ y2
    y1 = optcode(xs...)
    y2 = bounding_contract(AllConfigs{1}(), optcode, xs, BitArray(fill(true)), xs)
    @test y1 ≈ y2
end
