using GenericTensorNetworks, Test, Graphs

@testset "open pit mining" begin
    rewards = zeros(Int,6,6)
    rewards[1,:] .= [-4,-7,-7,-17,-7,-26]
    rewards[2,2:end-1] .= [39, -7, -7, -4]
    rewards[3,3:end-2] .= [1, 8]
    problem = OpenPitMining(rewards)
    res = solve(problem, SingleConfigMax())[]
    @test is_valid_mining(rewards, res.c.data)
    @test res.n == 21
    print_mining(rewards, res.c.data)
end