# # Binary Paint Shop Problem
#
# ## Overview
# The Binary Paint Shop Problem involves a sequence of cars, each appearing exactly twice.
# Each car must be painted red in one occurrence and blue in the other. The goal is to
# minimize the number of color changes when processing the sequence in order.
#
# This example demonstrates:
# * Formulating the paint shop problem
# * Converting it to a tensor network
# * Finding optimal coloring sequences
# * Visualizing solutions
#
# We'll use a character sequence where each character represents a car.

using GenericTensorNetworks, Graphs, GenericTensorNetworks.ProblemReductions

# Define our sequence (each character appears exactly twice)
sequence = collect("iadgbeadfcchghebif")

# ## Problem Visualization
# We can represent this problem as a graph:
# * Vertices are positions in the sequence
# * Blue edges connect the same car's two occurrences
# * Black edges connect adjacent positions in the sequence

rot(a, b, θ) = cos(θ)*a + sin(θ)*b, cos(θ)*b - sin(θ)*a
locations = [rot(0.0, 100.0, -0.25π - 1.5*π*(i-0.5)/length(sequence)) for i=1:length(sequence)]
graph = path_graph(length(sequence))
for i=1:length(sequence) 
    j = findlast(==(sequence[i]), sequence)
    i != j && add_edge!(graph, i, j)
end
show_graph(graph, locations; texts=string.(sequence), format=:svg, edge_colors=
    [sequence[e.src] == sequence[e.dst] ? "blue" : "black" for e in edges(graph)])

# Note: Vertices connected by blue edges must have different colors,
# and our goal becomes a min-cut problem on the black edges.

# ## Tensor Network Formulation
# Define the binary paint shop problem:
pshop = PaintShop(sequence)

# The objective is to minimize color changes:
objectives(pshop)

# Convert to tensor network representation:
problem = GenericTensorNetwork(pshop)

# ## Mathematical Background
# For each car $c_i$, we assign a boolean variable $s_{c_i} \in \{0,1\}$, where:
# -  $0$ means the first appearance is colored red
# -  $1$ means the first appearance is colored blue
#
# For adjacent positions $(i,i+1)$, we define edge tensors:
#
# 1. If both cars are at their first or both at their second appearance:
#    ```math
#    B^{\text{parallel}} = \begin{pmatrix}
#        x & 1 \\
#        1 & x
#    \end{pmatrix}
#    ```
#
#    (Cars tend to have the same configuration to avoid color changes)
#
# 2. Otherwise (one first, one second appearance):
#    ```math
#    B^{\text{anti-parallel}} = \begin{pmatrix}
#        1 & x \\
#        x & 1
#    \end{pmatrix}
#    ```
#
#    (Cars tend to have different configurations to avoid color changes)
# ## Solution Analysis
# ### 1. Paint Shop Polynomial
# The paint shop polynomial $P(G,x) = \sum_i p_i x^i$ counts colorings by number of color changes,
# where $p_i$ is the number of colorings with $(2m-1-i)$ color changes
paint_polynomial = solve(problem, GraphPolynomial())[]

# ### 2. Optimal Coloring Configurations
# Find all optimal coloring configurations:
best_configs = solve(problem, ConfigsMin())[]

# Note: We get two identical bitstrings corresponding to different vertex configurations
# due to bit-flip symmetry (we can start with either red or blue)

# ### 3. Solution Visualization
# Convert the optimal configuration to a coloring sequence:
painting1 = ProblemReductions.paint_shop_coloring_from_config(pshop, read_config(best_configs)[1])

# Visualize the optimal coloring:
show_graph(graph, locations; format=:svg, texts=string.(sequence),
    edge_colors=[sequence[e.src] == sequence[e.dst] ? "blue" : "black" for e in edges(graph)],
    vertex_colors=[isone(c) ? "red" : "black" for c in painting1], 
    config=GraphDisplayConfig(;vertex_text_color="white"))

# ### 4. Verification
# Verify the solution by counting the number of color switches:
num_paint_shop_color_switch(sequence, painting1)

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
