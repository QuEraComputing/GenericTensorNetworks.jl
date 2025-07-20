"""
    save_configs(filename, data::ConfigEnumerator; format=:binary)

Save configurations `data` to file `filename`. The format is `:binary` or `:text`.
"""
function save_configs(filename, data::ConfigEnumerator{N,S,C}; format::Symbol=:binary) where {N,S,C}
    if format == :binary
        write(filename, raw_matrix(data))
    elseif format == :text
        writedlm(filename, plain_matrix(data))
    else
        error("format must be `:binary` or `:text`, got `:$format`")
    end
end

"""
    load_configs(filename; format=:binary, bitlength=nothing, num_flavors=2)

Load configurations from file `filename`. The format is `:binary` or `:text`.
If the format is `:binary`, the bitstring length `bitlength` must be specified,
`num_flavors` specifies the degree of freedom.
"""
function load_configs(filename; bitlength=nothing, format::Symbol=:binary, num_flavors=2)
    if format == :binary
        bitlength === nothing && error("you need to specify `bitlength` for reading configurations from binary files.")
        S = ceil(Int, log2(num_flavors))
        C = _nints(bitlength, S)
        return _from_raw_matrix(StaticElementVector{bitlength,S,C}, reshape(reinterpret(UInt64, read(filename)),C,:))
    elseif format == :text
        return from_plain_matrix(readdlm(filename); num_flavors=num_flavors)
    else
        error("format must be `:binary` or `:text`, got `:$format`")
    end
end

function raw_matrix(x::ConfigEnumerator{N,S,C}) where {N,S,C}
    m = zeros(UInt64, C, length(x))
    @inbounds for i=1:length(x), j=1:C
        m[j,i] = x.data[i].data[j]
    end
    return m
end
function plain_matrix(x::ConfigEnumerator{N,S,C}) where {N,S,C}
    m = zeros(UInt8, N, length(x))
    @inbounds for i=1:length(x), j=1:N
        m[j,i] = x.data[i][j]
    end
    return m
end

function from_raw_matrix(m; bitlength, num_flavors=2)
    S = ceil(Int,log2(num_flavors))
    C = size(m, 1)
    T = StaticElementVector{bitlength,S,C}
    @assert bitlength*S <= C*64
    _from_raw_matrix(T, m)
end
function _from_raw_matrix(::Type{StaticElementVector{N,S,C}}, m::AbstractMatrix) where {N,S,C}
    data = zeros(StaticElementVector{N,S,C}, size(m, 2))
    @inbounds for i=1:size(m, 2)
        data[i] = StaticElementVector{N,S,C}(NTuple{C,UInt64}(view(m,:,i)))
    end
    return ConfigEnumerator(data)
end
function from_plain_matrix(m::Matrix; num_flavors=2)
    S = ceil(Int,log2(num_flavors))
    N = size(m, 1)
    C = _nints(N, S)
    T = StaticElementVector{N,S,C}
    _from_plain_matrix(T, m)
end
function _from_plain_matrix(::Type{StaticElementVector{N,S,C}}, m::AbstractMatrix) where {N,S,C}
    data = zeros(StaticElementVector{N,S,C}, size(m, 2))
    @inbounds for i=1:size(m, 2)
        data[i] = convert(StaticElementVector{N,S,C}, view(m, :, i))
    end
    return ConfigEnumerator(data)
end

# convert to Matrix
Base.Matrix(ce::ConfigEnumerator) = plain_matrix(ce)
Base.Vector(ce::StaticElementVector) = collect(ce)

########## saving tree ####################
"""
    save_sumproduct(filename, t::SumProductTree)

Serialize a sum-product tree into a file.
"""
save_sumproduct(filename::String, t::SumProductTree) = serialize(filename, dict_serialize_tree!(t, Dict{UInt,Any}()))

"""
    load_sumproduct(filename)

Deserialize a sum-product tree from a file.
"""
load_sumproduct(filename::String) = dict_deserialize_tree(deserialize(filename)...)

function dict_serialize_tree!(t::SumProductTree, d::Dict)
    id = objectid(t)
    if !haskey(d, id)
        if t.tag === GenericTensorNetworks.LEAF || t.tag === GenericTensorNetworks.ZERO || t.tag == GenericTensorNetworks.ONE
            d[id] = t
        else
            d[id] = (t.tag, objectid(t.left), objectid(t.right))
            dict_serialize_tree!(t.left, d)
            dict_serialize_tree!(t.right, d)
        end
    end
    return id, d
end

function dict_deserialize_tree(id::UInt, d::Dict)
    @assert haskey(d, id)
    content = d[id]
    if content isa SumProductTree
        return content
    else
        (tag, left, right) = content
        t = SumProductTree(tag, dict_deserialize_tree(left, d), dict_deserialize_tree(right, d))
        d[id] = t
        return t
    end
end

"""
    save_tensor_network(tn::GenericTensorNetwork; folder::String)

Serialize a tensor network to disk for storage/reloading. Creates three structured files:
- `code.json`: OMEinsum contraction code (tree structure and contraction order)
- `fixedvertices.json`: JSON-serialized Dict of pinned vertex configurations
- `problem.json`: Problem specification using ProblemReductions serialization

The target folder will be created recursively if it doesn't exist. Files are overwritten
if they already exist. Uses JSON for human-readable serialization with type preservation.

The saved files can be loaded using [`load_tensor_network`](@ref).

# Arguments
- `tn::GenericTensorNetwork`: a [`GenericTensorNetwork`](@ref) instance to serialize. Must contain valid code, problem, and fixedvertices fields.
- `folder::String`: Destination directory path. Parent directories will be created as needed.
"""
function save_tensor_network(tn::GenericTensorNetwork; folder::String)
    !isdir(folder) && mkpath(folder)

    OMEinsum.writejson(joinpath(folder, "code.json"), tn.code)
    
    open(joinpath(folder, "fixedvertices.json"), "w") do io
        JSON.print(io, tn.fixedvertices, 2)
    end
    
    ProblemReductions.writejson(joinpath(folder, "problem.json"), tn.problem)
    return nothing
end

"""
    load_tensor_network(folder::String) -> GenericTensorNetwork

Load a tensor network from disk that was previously saved using [`save_tensor_network`](@ref).
Reconstructs the network from three required files: contraction code, fixed vertices mapping, and problem specification.

# Arguments
- `folder::String`: Path to directory containing saved network files. Must contain:
  - `code.json`: Contraction order/structure from OMEinsum
  - `fixedvertices.json`: Dictionary of pinned vertex states
  - `problem.json`: Problem specification and parameters

# Returns
- `GenericTensorNetwork`: Reconstructed tensor network.
"""
function load_tensor_network(folder::String)
    !isdir(folder) && throw(SystemError("Folder not found: $folder"))
    
    code_path = joinpath(folder, "code.json")
    fixed_path = joinpath(folder, "fixedvertices.json")
    problem_path = joinpath(folder, "problem.json")
    
    !isfile(code_path) && throw(SystemError("Code file not found: $code_path"))
    !isfile(fixed_path) && throw(SystemError("Fixedvertices file not found: $fixed_path"))
    !isfile(problem_path) && throw(SystemError("Problem file not found: $problem_path"))
    
    code = OMEinsum.readjson(code_path)
    
    fixed_dict = JSON.parsefile(fixed_path)
    fixedvertices = Dict{labeltype(code),Int}(parse(Int, k) => v for (k, v) in fixed_dict)
    
    problem = ProblemReductions.readjson(problem_path)
    
    return GenericTensorNetwork(problem, code, fixedvertices)
end

