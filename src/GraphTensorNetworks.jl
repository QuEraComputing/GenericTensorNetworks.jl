module GraphTensorNetworks

using OMEinsumContractionOrders: SlicedEinsum
using Core: Argument
using TropicalNumbers
using OMEinsum
using OMEinsum: timespace_complexity, getixsv
using Graphs

export timespace_complexity, @ein_str
export GreedyMethod, TreeSA, SABipartite, KaHyParBipartite, MergeVectors, MergeGreedy

project_relative_path(xs...) = normpath(joinpath(dirname(dirname(pathof(@__MODULE__))), xs...))

include("utils.jl")
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
