module GenericTensorNetworks

using Core: Argument
using TropicalNumbers
using OMEinsum
using OMEinsum: contraction_complexity, timespace_complexity, timespacereadwrite_complexity, getixsv, NestedEinsum, getixs, getiy, DynamicEinCode
using Graphs, Random
using DelimitedFiles, Serialization, Printf
using LuxorGraphPlot
using LuxorGraphPlot.Luxor.Colors: @colorant_str
using LuxorGraphPlot: Layered
import Polynomials
using Polynomials: Polynomial, LaurentPolynomial, printpoly, fit
using FFTW
using Primes
using DocStringExtensions
using Base.Cartesian
using ProblemReductions
import ProblemReductions: ConstraintSatisfactionProblem, AbstractSatisfiabilityProblem, UnitWeight
import ProblemReductions: @bv_str, StaticElementVector, StaticBitVector, onehotv, _nints, hamming_distance
import ProblemReductions: is_set_covering, is_vertex_coloring, is_set_packing, is_dominating_set, is_matching, is_maximal_independent_set, cut_size, is_independent_set
import ProblemReductions: num_paint_shop_color_switch, spin_glass_from_matrix, CNF, CNFClause, BoolVar, satisfiable, @bools, ∨, ¬, ∧
import ProblemReductions: flavors, set_weights, weights, num_flavors, variables
import AbstractTrees: children, printnode, print_tree
import StatsBase

# OMEinsum
export timespace_complexity, timespacereadwrite_complexity, contraction_complexity, @ein_str, getixsv, getiyv
export GreedyMethod, TreeSA, SABipartite, KaHyParBipartite, MergeVectors, MergeGreedy

# estimate memory
export estimate_memory

# Algebras
export StaticBitVector, StaticElementVector, @bv_str, hamming_distance
export is_commutative_semiring
export Max2Poly, TruncatedPoly, Polynomial, LaurentPolynomial, Tropical, CountingTropical, StaticElementVector, Mod
export ConfigEnumerator, onehotv, ConfigSampler, SumProductTree
export CountingTropicalF64, CountingTropicalF32, TropicalF64, TropicalF32, ExtendedTropical
export generate_samples, OnehotVec

# Graphs
export random_regular_graph, diagonal_coupled_graph
export square_lattice_graph, unit_disk_graph, random_diagonal_coupled_graph, random_square_lattice_graph
export line_graph

# Tensor Networks (Graph problems)
export GenericTensorNetwork, optimize_code, UnitWeight
export flavors, variables, num_flavors, weights, fixedvertices, set_weights, energy_terms
export IndependentSet, mis_compactify!, is_independent_set
export MaximalIS, is_maximal_independent_set
export cut_size, MaxCut
export spinglass_energy, spin_glass_from_matrix, SpinGlass
export PaintShop, paintshop_from_pairs, num_paint_shop_color_switch, paint_shop_coloring_from_config, paint_shop_from_pairs
export Coloring, is_vertex_coloring
export Satisfiability, CNF, CNFClause, BoolVar, satisfiable, @bools, ∨, ¬, ∧
export DominatingSet, is_dominating_set
export Matching, is_matching
export SetPacking, is_set_packing
export SetCovering, is_set_covering

# Interfaces
export solve, SizeMax, SizeMin, PartitionFunction, CountingAll, CountingMax, CountingMin, GraphPolynomial, SingleConfigMax, SingleConfigMin, ConfigsAll, ConfigsMax, ConfigsMin, Single, AllConfigs

# Utilities
export save_configs, load_configs, hamming_distribution, save_sumproduct, load_sumproduct

# Readers
export read_size, read_count, read_config, read_size_count, read_size_config

# Visualization
export show_graph, show_configs, show_einsum, GraphDisplayConfig, render_locs, show_landscape
export AbstractLayout, SpringLayout, StressLayout, SpectralLayout, Layered, LayeredSpringLayout, LayeredStressLayout

project_relative_path(xs...) = normpath(joinpath(dirname(dirname(pathof(@__MODULE__))), xs...))

# Mods.jl fixed to v1.3.4
include("Mods.jl/src/Mods.jl")
using .Mods

include("utils.jl")
include("arithematics.jl")
include("networks.jl")
include("graph_polynomials.jl")
include("configurations.jl")
include("graphs.jl")
include("bounding.jl")
include("fileio.jl")
include("interfaces.jl")
include("deprecate.jl")
include("multiprocessing.jl")
include("visualize.jl")

end
