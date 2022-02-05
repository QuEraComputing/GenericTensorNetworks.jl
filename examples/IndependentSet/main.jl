# # Independent set problem

# ## Problem definition
# An independent set is defined in the monadic second order language as
# ```math
# \exists x_i,\ldots,x_M\left[\wedge_{i\neq j} (x_i\neq x_j \wedge \neg \textbf{adj}(x_i, x_j))\right]
# ```

# ## Solving properties

using GraphTensorNetworks, Graphs

graph = Graphs.smallgraph(:petersen)

problem = Independence(graph; optimizer=TreeSA(sc_weight=1.0, ntrials=10, βs=0.01:0.1:15.0, niters=20, rw_weight=0.2));

# maximum independent set size
maximum_independent_set_size = solve(problem, "size max")

# all independent sets
count_all_independent_sets = solve(problem, "counting sum")

# counting maximum independent sets
count_maximum_independent_sets = solve(problem, "counting max")

# counting independent sets of max two sizes
count_max2_independent_sets = solve(problem, "counting max2")

# compute the independence polynomial
independence_polynomial = solve(problem, "counting all (finitefield)")

# find one MIS
max_config = solve(problem, "config max")

# enumerate all MISs
all_max_configs = solve(problem, "configs max (bounded)")

# enumerate all IS configurations
all_independent_sets = solve(problem, "configs all")