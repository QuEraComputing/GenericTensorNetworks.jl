# References
## Graph problems
```@docs
solve
GenericTensorNetwork
GraphProblem
IndependentSet
MaximalIS
Matching
Coloring
DominatingSet
SpinGlass
MaxCut
PaintShop
Satisfiability
SetCovering
SetPacking
OpenPitMining
```

#### Graph Problem Interfaces

To subtype [`GraphProblem`](@ref), a new type must contain a `code` field to represent the (optimized) tensor network.
Interfaces [`GenericTensorNetworks.generate_tensors`](@ref), [`labels`](@ref), [`flavors`](@ref) and [`get_weights`](@ref) are required.
[`nflavor`](@ref) is optional.

```@docs
GenericTensorNetworks.generate_tensors
labels
energy_terms
flavors
get_weights
chweights
nflavor
fixedvertices
```

#### Graph Problem Utilities
```@docs
is_independent_set
is_maximal_independent_set
is_dominating_set
is_vertex_coloring
is_matching
is_set_covering
is_set_packing

cut_size
spinglass_energy
num_paint_shop_color_switch
paint_shop_coloring_from_config
mis_compactify!

CNF
CNFClause
BoolVar
satisfiable
@bools
∨
¬
∧

is_valid_mining
print_mining
```

## Properties
```@docs
PartitionFunction
SizeMax
SizeMin
CountingAll
CountingMax
CountingMin
GraphPolynomial
SingleConfigMax
SingleConfigMin
ConfigsAll
ConfigsMax
ConfigsMin
```

## Element Algebras
```@docs
is_commutative_semiring
```

```@docs
TropicalNumbers.Tropical
TropicalNumbers.CountingTropical
ExtendedTropical
GenericTensorNetworks.Mods.Mod
TruncatedPoly
Max2Poly
ConfigEnumerator
SumProductTree
ConfigSampler
```

`GenericTensorNetworks` also exports the [`Polynomial`](https://juliamath.github.io/Polynomials.jl/stable/polynomials/polynomial/#Polynomial-2) and [`LaurentPolynomial`](https://juliamath.github.io/Polynomials.jl/stable/polynomials/polynomial/#Polynomials.LaurentPolynomial) types defined in package `Polynomials`.

```@docs
StaticBitVector
StaticElementVector
OnehotVec
save_configs
load_configs
save_sumproduct
load_sumproduct
@bv_str
onehotv

generate_samples
hamming_distribution
```

## Tensor Network
```@docs
optimize_code
getixsv
getiyv
contraction_complexity
estimate_memory
@ein_str
GreedyMethod
TreeSA
SABipartite
KaHyParBipartite
MergeVectors
MergeGreedy
```

## Others
#### Graph
```@docs
show_graph
show_configs
show_einsum
show_landscape
GraphDisplayConfig
AbstractLayout
SpringLayout
StressLayout
SpectralLayout
Layered
LayeredSpringLayout
LayeredStressLayout
render_locs

diagonal_coupled_graph
square_lattice_graph
unit_disk_graph
line_graph

random_diagonal_coupled_graph
random_square_lattice_graph
```

One can also use `random_regular_graph` and `smallgraph` in [Graphs](https://github.com/JuliaGraphs/Graphs.jl) to build special graphs.

#### Multiprocessing
```@docs
GenericTensorNetworks.SimpleMultiprocessing.multiprocess_run
```