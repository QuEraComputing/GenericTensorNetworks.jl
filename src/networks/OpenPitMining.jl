"""
    OpenPitMining{ET, CT<:AbstractEinsum} <: GraphProblem
    OpenPitMining(rewards; openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing,
                 fixedvertices=Dict())

The open pit mining problem.
This problem can be solved in polynomial time with the pseudoflow algorithm.

Positional arguments
-------------------------------
* `rewards` is a matrix of rewards.

Keyword arguments
-------------------------------
* `openvertices` specifies labels of the output tensor.
* `optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
* `fixedvertices` is a dict to specify the values of labels, where a value can be `0` (not mined) or `1` (mined)
* `openvertices` is a tuple of labels to specify the output tensor. Theses degree of freedoms will not be contracted.

Example
-----------------------------------
```jldoctest; setup=:(using GenericTensorNetworks)
julia> rewards =  [-4  -7  -7  -17  -7  -26;
         0  39  -7   -7  -4    0;
         0   0   1    8   0    0;
         0   0   0    0   0    0;
         0   0   0    0   0    0;
         0   0   0    0   0    0];

julia> gp = OpenPitMining(rewards);

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
struct OpenPitMining{ET, CT<:AbstractEinsum} <: GraphProblem
    code::CT
    rewards::Matrix{ET}
    blocks::Vector{Tuple{Int,Int}}  # non-zero locations
    fixedvertices::Dict{Tuple{Int,Int},Int}
end

function get_blocks(rewards)
    blocks = Tuple{Int,Int}[]
    for i=1:size(rewards, 1), j=i:size(rewards,2)-i+1
        push!(blocks, (i,j))
    end
    return blocks
end
 

function OpenPitMining(rewards::Matrix{ET}; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing, fixedvertices=Dict{Tuple{Int,Int},Int}()) where ET
    # compute block locations
    blocks = Tuple{Int,Int}[]
    depends = Pair{Tuple{Int,Int},Tuple{Int,Int}}[]
    for i=1:size(rewards, 1), j=i:size(rewards,2)-i+1
        push!(blocks, (i,j))
        if i!=1
            push!(depends, (i,j)=>(i-1,j-1))
            push!(depends, (i,j)=>(i-1,j))
            push!(depends, (i,j)=>(i-1,j+1))
        end
    end
    code = EinCode([[[block] for block in blocks]...,
        [[dep.first, dep.second] for dep in depends]...], collect(Tuple{Int,Int},openvertices))
    OpenPitMining(_optimize_code(code, uniformsize_fix(code, 2, fixedvertices), optimizer, simplifier), rewards, blocks, fixedvertices)
end

function mining_tensor(::Type{T}) where T
    t = ones(T,2,2)
    t[2,1] = zero(T)   # first one is mined, but the second one is not mined.
    return t
end

flavors(::Type{<:OpenPitMining}) = [0, 1]
get_weights(gp::OpenPitMining, i::Int) = [0, gp.rewards[gp.blocks[i]...]]
terms(gp::OpenPitMining) = [[r] for r in gp.blocks]
labels(gp::OpenPitMining) = gp.blocks
fixedvertices(gp::OpenPitMining) = gp.fixedvertices

# generate tensors
function generate_tensors(x::T, gp::OpenPitMining) where T
    nblocks = length(gp.blocks)
    nblocks == 0 && return []
    ixs = getixsv(gp.code)
    # we only add labels at vertex tensors
    return select_dims([
        add_labels!(Array{T}[Ref(x) .^ get_weights(gp, i) for i=1:nblocks], ixs[1:nblocks], labels(gp))...,
        Array{T}[mining_tensor(T) for ix in ixs[nblocks+1:end]]...
        ], ixs, fixedvertices(gp)
    )
end

"""
    is_valid_mining(rewards::AbstractMatrix, config)

Return true if `config` (a boolean mask for the feasible region) is a valid mining of `rewards`.
"""
function is_valid_mining(rewards::AbstractMatrix, config)
    blocks = get_blocks(rewards)
    assign = Dict(zip(blocks, config))
    for block in blocks
        if block[1] != 1 && !iszero(assign[block])
            if iszero(assign[(block[1]-1, block[2]-1)]) ||
                iszero(assign[(block[1]-1, block[2])]) ||
                iszero(assign[(block[1]-1, block[2]+1)])
                return false
            end
        end
    end
    return true
end

"""
    print_mining(rewards::AbstractMatrix, config)

Printing the mining solution in a colored REPL.
"""
function print_mining(rewards::AbstractMatrix{T}, config) where T
    k = 0
    for i=1:size(rewards, 1)
        for j=1:size(rewards, 2)
            if j >= i && j <= size(rewards,2)-i+1
                k += 1
                if T <: Integer
                    str = @sprintf " %6i " rewards[i,j]
                else
                    str = @sprintf " %6.2F " rewards[i,j]
                end
                if iszero(config[k])
                    printstyled(str; color = :red)
                else
                    printstyled(str; color = :green)
                end
            else
                str = @sprintf " %6s " "◼"
                printstyled(str; color = :black)
            end
        end
        println()
    end
end

function _open_pit_mining_branching!(rewards::AbstractMatrix{T}, mask::AbstractMatrix{Bool}, setmask::AbstractMatrix{Bool}, idx::Int) where T
    # find next
    idx < 1 && return zero(T)
    i, j = divrem(idx-1, size(mask, 2)) .+ 1 # row-wise!
    while i > j || size(mask, 1)-i+1 < j || setmask[i, j]  # skip not allowed or already decided
        idx -= 1
        idx < 1 && return zero(T)
        i, j = divrem(idx-1, size(mask, 2)) .+ 1 # row-wise!
    end
    if rewards[i, j] < 0 # do not mine!
        setmask[i, j] = true
        return _open_pit_mining_branching!(rewards, mask, setmask, idx-1)
    else
        _mask = copy(mask)
        _setmask = copy(setmask)
        # CASE 1: try mine current block
        # set mask
        reward0 = set_recur!(mask, setmask, rewards, i, j)
        reward1 = _open_pit_mining_branching!(rewards, mask, setmask, idx-1) + reward0

        # CASE 1: try do not mine current block
        # unset mask
        _setmask[i, j] = true
        reward2 = _open_pit_mining_branching!(rewards, _mask, _setmask, idx-1)
        
        # choose the right branch
        if reward2 > reward1
            copyto!(mask, _mask)
            copyto!(setmask, _setmask)
            return reward2
        else
            return reward1
        end
    end
end

function set_recur!(mask, setmask, rewards::AbstractMatrix{T}, i, j) where T
    reward = zero(T)
    for k=1:i
        start = max(1, j-(i-k))
        stop = min(size(mask, 2), j+(i-k))
        @inbounds for l=start:stop
            if !setmask[k,l]
                mask[k,l] = true
                setmask[k,l] = true
                reward += rewards[k,l]
            end
        end
    end
    return reward
end

"""
    open_pit_mining_branching(rewards::AbstractMatrix)

Solve the open pit mining problem with the naive branching algorithm.
NOTE: open pit mining can be solved in polynomial time, but this one runs in exponential time.
"""
function open_pit_mining_branching(rewards::AbstractMatrix{T}) where T
    idx = length(rewards)
    mask = falses(size(rewards))
    rewards = _open_pit_mining_branching!(rewards, mask, falses(size(rewards)), idx)
    return rewards, mask
end