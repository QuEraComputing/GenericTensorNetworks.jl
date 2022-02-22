# References
## Graph problems
```@docs
solve
GraphProblem
IndependentSet
MaximalIS
Matching
Coloring
MaxCut
PaintShop
```

```@docs
set_packing
```

#### Graph Problem Interfaces

To subtype [`GraphProblem`](@ref), a new type must contain a `code` field to represent the (optimized) tensor network.
Interfaces [`GraphTensorNetworks.generate_tensors`](@ref), [`symbols`](@ref), [`flavors`](@ref) and [`get_weights`](@ref) are required.
[`nflavor`](@ref) is optional.

```@docs
GraphTensorNetworks.generate_tensors
symbols
flavors
get_weights
nflavor
```

#### Graph Problem Utilities
```@docs
is_independent_set
mis_compactify!

is_maximal_independent_set

cut_size
cut_assign

num_paint_shop_color_switch
paint_shop_coloring_from_config

is_good_vertex_coloring
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
Mods.Mod
Polynomials.Polynomial
TruncatedPoly
Max2Poly
ConfigEnumerator
TreeConfigEnumerator
ConfigSampler
```

```@docs
StaticBitVector
StaticElementVector
save_configs
load_configs
@bv_str
onehotv
```

## Tensor Network
```@docs
optimize_code
getixsv
getiyv
timespace_complexity
timespacereadwrite_complexity
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
spring_layout

diagonal_coupled_graph
square_lattice_graph
unit_disk_graph
line_graph

random_diagonal_coupled_graph
random_square_lattice_graph
```

One can also use `random_regular_graph` and `smallgraph` in [Graphs](https://github.com/JuliaGraphs/Graphs.jl) to build special graphs.

#### Shortcuts
```@docs
max_size
max_size_count
```
