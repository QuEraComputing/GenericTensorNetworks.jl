"""
$TYPEDEF

The open pit mining problem.
This problem can be solved in polynomial time with the pseudoflow algorithm.

Positional arguments
-------------------------------
* `rewards` is a matrix of rewards.
* `blocks` are the locations of the blocks.

Example
-----------------------------------
```jldoctest; setup=:(using GenericTensorNetworks)
julia> rewards =  [-4  -7  -7  -17  -7  -26;
         0  39  -7   -7  -4    0;
         0   0   1    8   0    0;
         0   0   0    0   0    0;
         0   0   0    0   0    0;
         0   0   0    0   0    0];

julia> gp = GenericTensorNetwork(OpenPitMining(rewards));

julia> res = solve(gp, SingleConfigMax())[]
(21.0, ConfigSampler{12, 1, 1}(111000100000))ₜ

julia> is_valid_mining(rewards, res.c.data)
true

julia> print_mining(rewards, res.c.data)
     -4      -7      -7     -17      -7     -26 
      ◼      39      -7      -7      -4       ◼ 
      ◼       ◼       1       8       ◼       ◼ 
      ◼       ◼       ◼       ◼       ◼       ◼ 
      ◼       ◼       ◼       ◼       ◼       ◼ 
      ◼       ◼       ◼       ◼       ◼       ◼
```

You will the the mining is printed as green in an colored REPL.
"""
function mining_tensor(::Type{T}) where T
    t = ones(T,2,2)
    t[2,1] = zero(T)   # first one is mined, but the second one is not mined.
    return t
end

energy_terms(gp::OpenPitMining) = [[r] for r in gp.blocks]
energy_tensors(x::T, c::OpenPitMining) where T = [_pow.(Ref(x), get_weights(c, i)) for i=1:length(c.blocks)]
function extra_terms(gp::OpenPitMining)
    depends = Pair{Tuple{Int,Int},Tuple{Int,Int}}[]
    for i=1:size(gp.rewards, 1), j=i:size(gp.rewards,2)-i+1
        if i!=1
            push!(depends, (i,j)=>(i-1,j-1))
            push!(depends, (i,j)=>(i-1,j))
            push!(depends, (i,j)=>(i-1,j+1))
        end
    end
    return [[dep.first, dep.second] for dep in depends]
end
extra_tensors(::Type{T}, gp::OpenPitMining) where T = [mining_tensor(T) for _ in extra_terms(gp)]
labels(gp::OpenPitMining) = gp.blocks

# get_weights interface
get_weights(c::OpenPitMining) = [c.rewards[b...] for b in c.blocks]
get_weights(gp::OpenPitMining, i::Int) = [0, gp.rewards[gp.blocks[i]...]]