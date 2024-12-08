# References
## Constraint Satisfaction Problems
```@docs
solve
GenericTensorNetwork
ConstraintSatisfactionProblem
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
```

#### Constraint Satisfaction Problem Interfaces

To subtype [`ConstraintSatisfactionProblem`](@ref), a new type must contain a `code` field to represent the (optimized) tensor network.
Interfaces [`GenericTensorNetworks.generate_tensors`](@ref), [`flavors`](@ref) and [`weights`](@ref) are required.
[`num_flavors`](@ref) is optional.

```@docs
GenericTensorNetworks.generate_tensors
flavors
weights
set_weights
is_weighted
num_flavors
fixedvertices
```

#### Constraint Satisfaction Problem Utilities
```@docs
hard_constraints
is_satisfied
local_solution_spec
solution_size
energy_mode
LargerSizeIsBetter
SmallerSizeIsBetter
energy

is_independent_set
is_maximal_independent_set
is_dominating_set
is_vertex_coloring
is_matching
is_set_covering
is_set_packing

cut_size
num_paint_shop_color_switch

CNF
CNFClause
BoolVar
satisfiable
@bools
∨
¬
∧

mis_compactify!
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


For reading the properties from the above element types, one can use the following functions.

```@docs
read_size
read_count
read_config
read_size_count
read_size_config
```

The following functions are for saving and loading configurations.

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
Except the `SimpleGraph` defined in [Graphs](https://github.com/JuliaGraphs/Graphs.jl), `GenericTensorNetworks` also defines the following types and functions.

```@docs
HyperGraph
UnitDiskGraph

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
line_graph

random_diagonal_coupled_graph
random_square_lattice_graph
```

One can also use `random_regular_graph` and `smallgraph` in [Graphs](https://github.com/JuliaGraphs/Graphs.jl) to build special graphs.

#### Multiprocessing
```@docs
GenericTensorNetworks.SimpleMultiprocessing.multiprocess_run
```