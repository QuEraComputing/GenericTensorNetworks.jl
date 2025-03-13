# # Spin Glass Problem
#
# ## Overview
# Spin glasses are magnetic systems characterized by disordered interactions between spins.
# They represent a fundamental model in statistical physics with applications in optimization,
# neural networks, and complex systems. This example demonstrates:
#
# * Formulating spin glass problems on simple graphs and hypergraphs
# * Converting them to tensor networks
# * Finding ground states and excited states
# * Computing partition functions and energy distributions
#
# We'll explore both standard graphs and hypergraphs to showcase the versatility of the approach.

using GenericTensorNetworks, Graphs

# ## Part 1: Spin Glass on a Simple Graph
# ### Problem Definition
# A spin glass on a graph G=(V,E) is defined by the Hamiltonian:
# H = ∑_{ij∈E} J_{ij} s_i s_j + ∑_{i∈V} h_i s_i
#
# Where:
# * s_i ∈ {-1,1} are spin variables
# * J_{ij} are coupling strengths between spins
# * h_i are local field terms
#
# We use boolean variables n_i = (1-s_i)/2 in our implementation.

# Create a Petersen graph instance
graph = Graphs.smallgraph(:petersen)

# Define vertex layout for visualization
rot15(a, b, i::Int) = cos(2i*π/5)*a + sin(2i*π/5)*b, cos(2i*π/5)*b - sin(2i*π/5)*a
locations = [[rot15(0.0, 60.0, i) for i=0:4]..., [rot15(0.0, 30, i) for i=0:4]...]
show_graph(graph, locations; format=:svg)

# ### Tensor Network Formulation
# Define an anti-ferromagnetic spin glass with uniform couplings:
spinglass = SpinGlass(graph, fill(1, ne(graph)), zeros(Int, nv(graph)))

# The objective is to minimize the energy:
objectives(spinglass)

# Convert to tensor network representation:
problem = GenericTensorNetwork(spinglass)

# ### Mathematical Background
# The tensor network for a spin glass uses:
#
# 1. Edge Tensors: For edge $(i,j)$ with coupling $J_{ij}$:
#    ```math
#    B_{s_i,s_j}(x) = \begin{pmatrix}
#        x^{J_{ij}} & x^{-J_{ij}} \\
#        x^{-J_{ij}} & x^{J_{ij}}
#    \end{pmatrix}
#    ```
#    * Contributes $x^{J_{ij}}$ when spins are aligned ($s_i = s_j$)
#    * Contributes $x^{-J_{ij}}$ when spins are anti-aligned ($s_i ≠ s_j$)
#
# 2. Vertex Tensors: For vertex $i$ with local field $h_i$:
#    ```math
#    W_i(x) = \begin{pmatrix}
#        x^{h_i} \\
#        x^{-h_i}
#    \end{pmatrix}
#    ```
#
# This formulation allows efficient computation of various properties.
# ### Solution Analysis
# #### 1. Energy Extrema
# Find the minimum energy (ground state):
Emin = solve(problem, SizeMin())[]

# Find the maximum energy (highest excited state):
Emax = solve(problem, SizeMax())[]
# Note: The state with highest energy has all spins with the same value

# #### 2. Partition Function
# The graph polynomial $Z(G,J,h,x) = \sum_i c_i x^i$ counts configurations by energy,
# where $c_i$ is the number of configurations with energy $i$
partition_function = solve(problem, GraphPolynomial())[]

# #### 3. Ground State Configuration
# Find one ground state configuration:
ground_state = read_config(solve(problem, SingleConfigMin())[])

# Verify the energy matches our earlier computation:
Emin_verify = energy(problem.problem, ground_state)

# Visualize the ground state:
show_graph(graph, locations; vertex_colors=[
    iszero(ground_state[i]) ? "white" : "red" for i=1:nv(graph)], format=:svg)
# Note: Red vertices represent spins with value -1 (or 1 in boolean representation)

# ## Part 2: Spin Glass on a Hypergraph
# ### Problem Definition
# A spin glass on a hypergraph H=(V,E) is defined by the Hamiltonian:
# E = ∑_{c∈E} w_c ∏_{v∈c} S_v
#
# Where:
# * S_v ∈ {-1,1} are spin variables
# * w_c are coupling strengths for hyperedges
#
# We use boolean variables s_v = (1-S_v)/2 in our implementation.

# Define a hypergraph with 15 vertices
num_vertices = 15

hyperedges = [[1,3,4,6,7], [4,7,8,12], [2,5,9,11,13],
    [1,2,14,15], [3,6,10,12,14], [8,14,15], 
    [1,2,6,11], [1,2,4,6,8,12]]

weights = [-1, 1, -1, 1, -1, 1, -1, 1]

# Define the hypergraph spin glass problem:
hyperspinglass = SpinGlass(HyperGraph(num_vertices, hyperedges), weights, zeros(Int, num_vertices))

# Convert to tensor network representation:
hyperproblem = GenericTensorNetwork(hyperspinglass)

# ### Solution Analysis
# #### 1. Energy Extrema
# Find the minimum energy (ground state):
Emin = solve(hyperproblem, SizeMin())[]

# Find the maximum energy (highest excited state):
Emax = solve(hyperproblem, SizeMax())[]
# Note: Spin configurations can be chosen to make all hyperedges
# have either even or odd spin parity

# #### 2. Partition Function and Polynomial
# Compute the partition function at inverse temperature β = 2.0:
β = 2.0
Z = solve(hyperproblem, PartitionFunction(β))[]

# Compute the infinite temperature partition function (counts all configurations):
solve(hyperproblem, PartitionFunction(0.0))[]

# Compute the full graph polynomial:
poly = solve(hyperproblem, GraphPolynomial())[]

# #### 3. Ground State Configuration
# Find one ground state configuration:
ground_state = read_config(solve(hyperproblem, SingleConfigMin())[])

# Verify the energy matches our earlier computation:
Emin_verify = energy(hyperproblem.problem, ground_state)
# The result should match the Emin value computed earlier

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
