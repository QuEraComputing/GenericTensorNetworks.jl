module GraphTensorNetworks

using OMEinsumContractionOrders: OMEinsum
using Core: Argument
using TropicalGEMM, TropicalNumbers
using OMEinsum
using OMEinsum: timespace_complexity, collect_ixs
using LightGraphs

export timespace_complexity, @ein_str

project_relative_path(xs...) = normpath(joinpath(dirname(dirname(pathof(@__MODULE__))), xs...))

# patch to permutedims
using Base.Cartesian
using Base: size_to_strides, checkdims_perm
for (V, PT, BT) in Any[((:N,), BitArray, BitArray), ((:T,:N), Array, StridedArray)]
    @eval @generated function Base.permutedims!(P::$PT{$(V...)}, B::$BT{$(V...)}, perm) where $(V...)
        quote
            checkdims_perm(P, B, perm)

            #calculates all the strides
            native_strides = size_to_strides(1, size(B)...)
            strides_1 = 0
            @nexprs $N d->(strides_{d+1} = native_strides[perm[d]])

            #Creates offset, because indexing starts at 1
            offset = 1 - sum(@ntuple $N d->strides_{d+1})

            sumc = 0
            ind = 1
            @nexprs 1 d->(counts_{$N+1} = strides_{$N+1}) # a trick to set counts_($N+1)
            @nloops($N, i, P,
                    d->(df_d=i_d*strides_{d+1} ;sumc += df_d), # PRE
                    d->(sumc -= df_d), # POST
                    begin # BODY
                        @inbounds P[ind] = B[sumc+offset]
                        ind += 1
                    end)

            return P
        end
    end
end

include("bitvector.jl")
include("arithematics.jl")
include("networks.jl")
include("graph_polynomials.jl")
include("configurations.jl")
include("graphs.jl")
include("bounding.jl")
include("viz.jl")
include("interfaces.jl")

using Requires
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end
