# References
## Graph problems
```@docs
solve
GraphProblem
IndependentSet
MaximalIS
Matching
Coloring
DominatingSet
MaxCut
PaintShop
Satisfiability
```

```@docs
set_packing
```

#### Graph Problem Interfaces

To subtype [`GraphProblem`](@ref), a new type must contain a `code` field to represent the (optimized) tensor network.
Interfaces [`GraphTensorNetworks.generate_tensors`](@ref), [`labels`](@ref), [`flavors`](@ref) and [`get_weights`](@ref) are required.
[`nflavor`](@ref) is optional.

```@docs
GraphTensorNetworks.generate_tensors
labels
flavors
get_weights
nflavor
```

#### Graph Problem Utilities
```@docs
is_independent_set
mis_compactify!

is_maximal_independent_set

is_dominating_set

cut_size

num_paint_shop_color_switch
paint_shop_coloring_from_config

is_good_vertex_coloring

is_matching

CNF
CNFClause
BoolVar
satisfiable
@bools
∨
¬
∧
```

## Properties
```@docs
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
Mods.Mod
TruncatedPoly
Max2Poly
ConfigEnumerator
SumProductTree
ConfigSampler
```

`GraphTensorNetworks` also exports the [`Polynomial`](https://juliamath.github.io/Polynomials.jl/stable/polynomials/polynomial/#Polynomial-2) type defined in package `Polynomials`.

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
timespace_complexity
timespacereadwrite_complexity
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
show_gallery
spring_layout

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
GraphTensorNetworks.SimpleMultiprocessing.multiprocess_run
```

#### Shortcuts
```@docs
max_size
max_size_count
```
