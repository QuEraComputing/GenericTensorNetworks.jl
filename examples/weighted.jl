# # Weighted problems
# Let us use the maximum independent set problem on Petersen graph as an example.

using GenericTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

# The following code computes the weighted MIS problem.
problem = IndependentSet(graph; weights=collect(1:10));

# `weights` is a key word argument that can be a vector for weighted graphs or a `NoWeight()` object for unweighted graphs. The maximum independent set can be found as follows.

max_config_weighted = solve(problem, SingleConfigMax())[]

# Let us visualize the solution.
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a

locations = [[rot15(0.0, 1.0, i) for i=0:4]..., [rot15(0.0, 0.6, i) for i=0:4]...]

show_graph(graph; locs=locations, vertex_colors=
          [iszero(max_config_weighted.c.data[i]) ? "white" : "red" for i=1:nv(graph)])

# For weighted MIS problem, a property that many people care about is the "energy spectrum", or the largest weights.
# We just feed a positional argument in the [`SizeMax`](@ref) constructor as the number of largest weights.
spectrum = solve(problem, SizeMax(10))[]

# The return value has type [`ExtendedTropical`](@ref), which contains one field `orders`. The `orders` is a vector of [`Tropical`](@ref) numbers.
spectrum.orders

# We can get weighted independent sets with maximum 5 sizes.
max5_configs = solve(problem, SingleConfigMax(5))[]

# The return value also has type [`ExtendedTropical`](@ref), but this time the element type of `orders` has been changed to [`CountingTropical`](@ref)`{Float64,`[`ConfigSampler`](@ref)`}`.
max5_configs.orders

# Let us visually check these configurations
show_gallery(graph, (1, 5); locs=locations, vertex_configs=[max5_configs.orders[k].c.data for k=1:5])

