# References
## Graph problems
```@docs
solve
GraphProblem
Independence
MaximalIndependence
Matching
Coloring
MaxCut
PaintShop
```

```@docs
set_packing
```

#### Graph Problem Interfaces
```@docs
generate_tensors
symbols
flavors
get_weights
nflavor
```

To subtype [`GraphProblem`](@ref), the new type must contain a `code` field to represent the (optimized) tensor network.
Interfaces [`generate_tensors`](@ref), [`symbols`](@ref) and [`flavors`](@ref) are required.
[`get_weights`] and [`nflavor`] are optimal.


## Properties
```@docs
SizeMax
CountingAll
CountingMax
GraphPolynomial
SingleConfigMax
ConfigsAll
ConfigsMax
```

## Element Algebras
```@docs
TropicalNumbers.Tropical
TropicalNumbers.CountingTropical
Mods.Mod
Polynomials.Polynomial
TruncatedPoly
Max2Poly
ConfigEnumerator
ConfigSampler
```

```@docs
StaticBitVector
StaticElementVector
save_configs
load_configs
@bv_str
onehotv
is_commutative_semiring
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
is_independent_set
mis_compactify!
show_graph

diagonal_coupled_graph
square_lattice_graph
unitdisk_graph
```

One can also use `random_regular_graph` and `smallgraph` in [Graphs](https://github.com/JuliaGraphs/Graphs.jl) to build special graphs.

#### Lower level APIs
```@docs
best_solutions
best2_solutions
solutions
all_solutions
graph_polynomial
max_size
max_size_count
```