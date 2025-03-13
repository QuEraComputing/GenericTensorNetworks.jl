# # Set Covering Problem
#
# ## Overview
# The Set Covering Problem is a fundamental optimization challenge: given a collection of sets,
# find the minimum number of sets needed to cover all elements. This NP-hard problem appears
# in many real-world applications including facility location, scheduling, and resource allocation.
#
# This example demonstrates:
# * Formulating a set covering problem
# * Converting it to a tensor network
# * Finding minimum set covers
# * Analyzing the solution space
#
# We'll use the camera location and stadium area example from the Cornell University
# Computational Optimization Open Textbook.

using GenericTensorNetworks, Graphs

# Define our sets (each representing a camera's coverage area)
sets = [[1,3,4,6,7], [4,7,8,12], [2,5,9,11,13],
    [1,2,14,15], [3,6,10,12,14], [8,14,15], 
    [1,2,6,11], [1,2,4,6,8,12]]

# ## Tensor Network Formulation
# Define the set covering problem:
setcover = SetCovering(sets)

# The problem consists of:
# 1. Coverage constraints: Every element must be covered by at least one set
# 2. Optimization objective: Minimize the number of sets used
constraints(setcover)
#
objectives(setcover)

# Convert to tensor network representation:
problem = GenericTensorNetwork(setcover)

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
#        0 & \text{if all } s_i = 0 \text{ (element not covered - invalid)} \\
#        1 & \text{otherwise (element is covered - valid)}
#    \end{cases}
#    ```
#
# Check the contraction complexity:
contraction_complexity(problem)

# ## Solution Analysis
# ### 1. Set Covering Polynomial
# The polynomial $P(S,x) = \sum_i c_i x^i$ counts set covers by size,
# where $c_i$ is the number of valid covers using $i$ sets
covering_polynomial = solve(problem, GraphPolynomial())[]

# ### 2. Minimum Set Cover Size
# Find the minimum number of sets needed:
min_cover_size = solve(problem, SizeMin())[]

# Count minimum set covers:
counting_minimum_setcovering = solve(problem, CountingMin())[]

# ### 3. Minimum Set Cover Configurations
# Enumerate all minimum set covers:
min_configs = read_config(solve(problem, ConfigsMin())[])

# The two optimal solutions are $\{z_1, z_3, z_5, z_6\}$ and $\{z_2, z_3, z_4, z_5\}$,
# where $z_i$ represents the $i$-th set in our original list.

# Verify solutions are valid:
all(c->is_set_covering(problem.problem, c), min_configs)

# Note: For finding just one minimum set cover, use the SingleConfigMin property

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
