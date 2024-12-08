# Updates in v3.0

1. The solution size of `Coloring`/`Satisfiability` is now defined as the number of violations of colors/clauses. The smaller the better now.
2. Rename `best_solutions` to `largest_solutions`, `best2_solutions` to `largest2_solutions` and `bestk_solutions` to `largestk_solutions`.
3. Remove the weights from `PaintShop`.
4. Remove the weights on vertices from `MaxCut`.
5. `SpinGlass` is no longer specified by cliques. It is now specified by graphs or hypergraphs. Weights can be defined on both edges and vertices.
6. Remove `unit_disk_graph`, replace it with `UnitDiskGraph` from `ProblemReductions`.