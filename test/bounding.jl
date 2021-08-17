using Test, OMEinsum, GraphTensorNetworks, TropicalNumbers, Random
using GraphTensorNetworks: cached_einsum, generate_masktree, masked_einsum, CacheTree

@testset "cached einsum" begin
    xs = map(x->Tropical.(x), [randn(2,2), randn(2), randn(2,2), randn(2,2), randn(2,2)])
    code = ein"((ij,j),jk, kl), ii->kli"
    size_dict = uniformsize(code, 2)
    c = cached_einsum(code, xs, size_dict)
    @test c.content == code(xs...)
    mt = generate_masktree(code, c, rand(Bool,2,2,2), size_dict)
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
        y2 = bounding_contract(code, xs, BitArray(ones(Bool,2,2,2)), xs)
        @test y1 ≈ y2
    end
    rawcode = Independence(random_regular_graph(10, 3)).code
    optcode = OMEinsum.optimize_greedy(rawcode, uniformsize(rawcode, 2))
    xs = map(OMEinsum.getixs(rawcode)) do ix
        length(ix)==1 ? GraphTensorNetworks.misv(TropicalF64(1.0)) : GraphTensorNetworks.misb(TropicalF64)
    end
    y1 = rawcode(xs...)
    y2 = bounding_contract(rawcode, xs, BitArray(fill(true)), xs)
    @test y1 ≈ y2
    y1 = optcode(xs...)
    y2 = bounding_contract(optcode, xs, BitArray(fill(true)), xs)
    @test y1 ≈ y2
end

