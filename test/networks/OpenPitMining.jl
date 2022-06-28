using GenericTensorNetworks, Test, Graphs

@testset "open pit mining" begin
    rewards = zeros(Int,6,6)
    rewards[1,:] .= [-4,-7,-7,-17,-7,-26]
    rewards[2,2:end-1] .= [39, -7, -7, -4]
    rewards[3,3:end-2] .= [1, 8]
    problem = OpenPitMining(rewards)
    @test get_weights(problem) == [-4,-7,-7,-17,-7,-26, 39, -7, -7, -4, 1, 8]
    @test get_weights(chweights(problem, fill(3, 20))) == fill(3, 12)
    res = solve(problem, SingleConfigMax())[]
    @test is_valid_mining(rewards, res.c.data)
    @test res.n == 21
    print_mining(rewards, res.c.data)
    val, mask = GenericTensorNetworks.open_pit_mining_branching(rewards)
    @test val == res.n
    res_b = map(block->mask[block...], problem.blocks)
    @test res_b == [res.c.data...]
end