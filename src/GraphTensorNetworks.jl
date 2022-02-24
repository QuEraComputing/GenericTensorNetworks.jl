module GraphTensorNetworks

using OMEinsumContractionOrders: SlicedEinsum
using Core: Argument
using TropicalNumbers
using OMEinsum
using OMEinsum: timespace_complexity, getixsv
using Graphs

# OMEinsum
export timespace_complexity, timespacereadwrite_complexity, @ein_str, getixsv, getiyv
export GreedyMethod, TreeSA, SABipartite, KaHyParBipartite, MergeVectors, MergeGreedy

# Algebras
export StaticBitVector, StaticElementVector, @bv_str
export is_commutative_semiring
export Max2Poly, TruncatedPoly, Polynomial, Tropical, CountingTropical, StaticElementVector, Mod, ConfigEnumerator, onehotv, ConfigSampler, TreeConfigEnumerator
export CountingTropicalF64, CountingTropicalF32, TropicalF64, TropicalF32

# Lower level APIs
export AllConfigs, SingleConfig
export best_solutions, best2_solutions, solutions, all_solutions
export bestk_solutions
export contractx, graph_polynomial, max_size, max_size_count

# Graphs
export random_regular_graph, diagonal_coupled_graph, is_independent_set, is_maximal_independent_set
export square_lattice_graph, unit_disk_graph, random_diagonal_coupled_graph, random_square_lattice_graph
export line_graph

# Tensor Networks (Graph problems)
export GraphProblem, IndependentSet, MaximalIS, Matching, 
    Coloring, optimize_code, set_packing, MaxCut, PaintShop,
    paintshop_from_pairs, UnWeighted, Satisfiability
export flavors, labels, terms, nflavor, get_weights
export mis_compactify!, cut_size, num_paint_shop_color_switch, paint_shop_coloring_from_config
export is_good_vertex_coloring
export CNF, CNFClause, BoolVar, satisfiable, @bools, ∨, ¬, ∧

# Interfaces
export solve, SizeMax, SizeMin, CountingAll, CountingMax, CountingMin, GraphPolynomial, SingleConfigMax, SingleConfigMin, ConfigsAll, ConfigsMax, ConfigsMin

# Utilities
export save_configs, load_configs

# Visualization
export show_graph, spring_layout

project_relative_path(xs...) = normpath(joinpath(dirname(dirname(pathof(@__MODULE__))), xs...))

include("utils.jl")
include("bitvector.jl")
include("arithematics.jl")
include("networks/networks.jl")
include("graph_polynomials.jl")
include("configurations.jl")
include("graphs.jl")
include("bounding.jl")
include("visualize.jl")
include("interfaces.jl")
include("deprecate.jl")

using Requires
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end
