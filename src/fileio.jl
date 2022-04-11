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
    load_configs(filename; format=:binary, bitlength=nothing, nflavors=2)

Load configurations from file `filename`. The format is `:binary` or `:text`.
If the format is `:binary`, the bitstring length `bitlength` must be specified,
`nflavors` specifies the degree of freedom.
"""
function load_configs(filename; bitlength=nothing, format::Symbol=:binary, nflavors=2)
    if format == :binary
        bitlength === nothing && error("you need to specify `bitlength` for reading configurations from binary files.")
        S = ceil(Int, log2(nflavors))
        C = _nints(bitlength, S)
        return _from_raw_matrix(StaticElementVector{bitlength,S,C}, reshape(reinterpret(UInt64, read(filename)),C,:))
    elseif format == :text
        return from_plain_matrix(readdlm(filename); nflavors=nflavors)
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

function from_raw_matrix(m; bitlength, nflavors=2)
    S = ceil(Int,log2(nflavors))
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
function from_plain_matrix(m::Matrix; nflavors=2)
    S = ceil(Int,log2(nflavors))
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

