# # Set Packing Problem
#
# ## Overview
# The Set Packing Problem is a generalization of the Independent Set problem from simple graphs
# to hypergraphs. Given a collection of sets, the goal is to find the maximum number of mutually
# disjoint sets (sets that share no common elements).
#
# This example demonstrates:
# * Formulating a set packing problem
# * Converting it to a tensor network
# * Finding maximum set packings
# * Analyzing the solution space
#
# We'll use the same sets from the [Set Covering Problem](@ref) example for comparison.

using GenericTensorNetworks, Graphs

# Define our sets (each representing a camera's coverage area)
sets = [[1,3,4,6,7], [4,7,8,12], [2,5,9,11,13],
    [1,2,14,15], [3,6,10,12,14], [8,14,15], 
    [1,2,6,11], [1,2,4,6,8,12]]

# ## Tensor Network Formulation
# Define the set packing problem:
setpacking = SetPacking(sets)

# The problem consists of:
# 1. Packing constraints: No element can be covered by more than one set
# 2. Optimization objective: Maximize the number of sets used
constraints(setpacking)
#
objectives(setpacking)

# Convert to tensor network representation:
problem = GenericTensorNetwork(setpacking)

# ## Mathematical Background
# For each set $s$ with weight $w_s$, we assign a boolean variable $v_s ∈ \{0,1\}$,
# indicating whether the set is included in the solution.
#
# The network uses two types of tensors:
#
# 1. Set Tensors: For each set $s$:
#    ```math
#    W(x_s, w_s) = \begin{pmatrix}
#        1 \\
#        x_s^{w_s}
#    \end{pmatrix}
#    ```
#
# 2. Element Constraint Tensors: For each element a and its containing sets N(a):
#    ```math
#    B_{s_1,...,s_{|N(a)|}} = 
#    \begin{cases}
#        0 & \text{if } \sum_i s_i > 1 \text{ (element covered multiple times - invalid)} \\
#        1 & \text{otherwise (element covered at most once - valid)}
#    \end{cases}
#    ```
#
# Check the contraction complexity:
contraction_complexity(problem)

# ## Solution Analysis
# ### 1. Set Packing Polynomial
# The polynomial $P(S,x) = \sum_i c_i x^i$ counts set packings by size,
# where $c_i$ is the number of valid packings using $i$ sets
packing_polynomial = solve(problem, GraphPolynomial())[]

# ### 2. Maximum Set Packing Size
# Find the maximum number of mutually disjoint sets:
max_packing_size = solve(problem, SizeMax())[]

# Count maximum set packings:
counting_maximum_set_packing = solve(problem, CountingMax())[]

# ### 3. Maximum Set Packing Configurations
# Enumerate all maximum set packings:
max_configs = read_config(solve(problem, ConfigsMax())[])

# The optimal solution is $\{z_1, z_3, z_6\}$ with size 3,
# where $z_i$ represents the $i$-th set in our original list.

# Verify solutions are valid:
all(c->is_set_packing(problem.problem, c), max_configs)

# Note: For finding just one maximum set packing, use the SingleConfigMax property

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.