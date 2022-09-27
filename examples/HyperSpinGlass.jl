# # Hyper-Spin-glass problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem definition
# The hyper-spin-glass problem is a natural generalization of the spin-glass problem from simple graph to hypergraph.
# A hyper-spin-glass problem of hypergraph ``H = (V, E)`` can be characterized by the following energy function
# ```math
# E = \sum_{c \in E} w_{c} \prod_{v\in c} S_v
# ```
# where ``S_v \in \{-1, 1\}``, ``w_c`` is coupling strength associated with hyperedge ``c``.
# In the program, we use boolean variable ``s_v = \frac{1-S_v}{2}`` to represent a spin configuration.

using GenericTensorNetworks

# In the following, we are going to defined an spin glass problem for the following hypergraph.
num_vertices = 15

hyperedges = [[1,3,4,6,7], [4,7,8,12], [2,5,9,11,13],
    [1,2,14,15], [3,6,10,12,14], [8,14,15], 
    [1,2,6,11], [1,2,4,6,8,12]]

weights = [-1, 1, -1, 1, -1, 1, -1, 1];

# The energy function is
# ```math
# \begin{align*}
# E = &-s_1s_3s_4s_6s_7 + s_4s_7s_8s_{12} - s_2s_5s_9s_{11}s_{13} +\\
#    &s_1s_2s_{14}s_{15} - s_3s_6s_{10}s_{12}s_{14} + s_8s_{14}s_{15} +\\
#    &s_1s_2s_6s_{11} - s_1s_2s_4s_6s_8s_{12}
# \end{align*}
# ```

# ## Generic tensor network representation
# We define an anti-ferromagnetic spin glass problem as
problem = HyperSpinGlass(num_vertices, hyperedges; weights);

# ### Theory (can skip)
# Let ``H = (V, E)`` be a hypergraph. The tensor network for the partition function of the energy model for ``H`` can be defined as the following triple of (alphabet of labels, input tensors, output labels).
# ```math
# \begin{cases}
# \Lambda &= \{s_v \mid v \in V\}\\
# \mathcal{T} &= \{B^{(c)}_{s_{N(c, 1),N(c, 2),\ldots,N(c, d(c))}} \mid c \in E\} \cup \{W^{(v)}_{s_v} \mid v \in V\}\\
# \sigma_o &= \varepsilon
# \end{cases}
# ```
# where ``s_v \in \{0, 1\}`` is the boolean degreen associated to vertex ``v``,
# ``N(c, k)`` is the ``k``th vertex of hyperedge ``c``, and ``d(c)`` is the degree of ``c``.
# The edge tensor ``B^{(c)}`` is defined as
# ```math
# B^{(c)} = \begin{cases}
# x^{w_c} &  (\sum_{v\in c} s_v) \;is\; even, \\
# x^{-w_c}  &  otherwise.
# \end{cases}
# ```
# and the vertex tensor ``W^{(v)}`` (used to carry labels) is defined as
# ```math
# W^{(v)} = \left(\begin{matrix}1_v\\ 1_v\end{matrix}\right)
# ```

# ## Solving properties
# ### Minimum and maximum energies
# Its ground state energy is -8.
Emin = solve(problem, SizeMin())[]
# While the state correspond to the highest energy has the ferromagnetic order.
Emax = solve(problem, SizeMax())[]

# In this example, the spin configurations can be chosen to make all hyperedges having even or odd spin parity.

# ### Counting properties
# ##### partition function and graph polynomial
# The graph polynomial defined for the hyper-spin-glass problem is a Laurent polynomial
# ```math
# Z(G, w, x) = \sum_{k=E_{\rm min}}^{E_{\rm max}} c_k x^k,
# ```
# where ``E_{\rm min}`` and ``E_{\rm max}`` are minimum and maximum energies,
# ``c_k`` is the number of spin configurations with energy ``k``.
# Let the inverse temperature ``\beta = 2``, the partition function is
β = 2.0
Z = solve(problem, PartitionFunction(β))[]

# The infinite temperature partition function is the counting of all feasible configurations
solve(problem, PartitionFunction(0.0))[]

# Let ``x = e^\beta``, it corresponds to the partition function of a spin glass at temperature ``\beta^{-1}``.
poly = solve(problem, GraphPolynomial())[]

# ### Configuration properties
# ##### finding a ground state
ground_state = solve(problem, SingleConfigMin())[].c.data

Emin_verify = hyperspinglass_energy(hyperedges, ground_state; weights)

# You should see a consistent result as above `Emin`.