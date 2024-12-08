# # Weighted problems
# Let us use the maximum independent set problem on Petersen graph as an example.

using GenericTensorNetworks, GenericTensorNetworks.ProblemReductions, Graphs

graph = Graphs.smallgraph(:petersen)

# The following code constructs a weighted MIS problem instance.
problem = GenericTensorNetwork(IndependentSet(graph, collect(1:10)));
GenericTensorNetworks.weights(problem)

# The tensor labels that associated with the weights can be accessed by
ProblemReductions.local_solution_spec(problem.problem)

# Here, the `weights` keyword argument can be a vector for weighted graphs or `UnitWeight()` for unweighted graphs.
# Most solution space properties work for unweighted graphs also work for the weighted graphs.
# For example, the maximum independent set can be found as follows.

max_config_weighted = solve(problem, SingleConfigMax())[]

# Let us visualize the solution.
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg, vertex_colors=
          [iszero(max_config_weighted.c.data[i]) ? "white" : "red" for i=1:nv(graph)])

# The only solution space property that can not be defined for general real-weighted (not including integer-weighted) graphs is the [`GraphPolynomial`](@ref).

# For the weighted MIS problem, a useful solution space property is the "energy spectrum", i.e. the largest several configurations and their weights.
# We can use the solution space property is [`SizeMax`](@ref)`(10)` to compute the largest 10 weights.
spectrum = solve(problem, SizeMax(10))[]

# The return value has type [`ExtendedTropical`](@ref), which contains one field `orders`.
spectrum.orders

# We can see the `order` is a vector of [`Tropical`](@ref) numbers.
# Similarly, we can get weighted independent sets with maximum 5 sizes as follows.
max5_configs = read_config(solve(problem, SingleConfigMax(5))[])

# The return value of `solve` has type [`ExtendedTropical`](@ref), but this time the element type of `orders` has been changed to [`CountingTropical`](@ref)`{Float64,`[`ConfigSampler`](@ref)`}`.
# Let us visually check these configurations
show_configs(graph, locations, [max5_configs[j] for i=1:1, j=1:5]; padding_left=20)