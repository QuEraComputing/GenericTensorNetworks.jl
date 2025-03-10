# # Boolean Satisfiability Problem
#
# ## Overview
# The Boolean Satisfiability Problem (SAT) determines whether there exists an assignment of 
# truth values to variables that makes a given Boolean formula evaluate to true.
# 
# This example demonstrates:
# * Formulating a SAT problem in Conjunctive Normal Form (CNF)
# * Converting it to a tensor network
# * Finding satisfying assignments
# * Counting satisfiable solutions
#
# We'll work with a CNF formula consisting of multiple clauses.

using GenericTensorNetworks, GenericTensorNetworks.ProblemReductions

# Define boolean variables
@bools a b c d e f g

# Create a CNF formula: (a ∨ b ∨ ¬d ∨ ¬e) ∧ (¬a ∨ d ∨ e ∨ ¬f) ∧ (f ∨ g) ∧ (¬b ∨ c)
cnf = ∧(∨(a, b, ¬d, ¬e), ∨(¬a, d, e, ¬f), ∨(f, g), ∨(¬b, c))

# ## Manual Verification
# For small problems, we can manually find and verify a satisfying assignment:
assignment = Dict([:a=>true, :b=>false, :c=>false, :d=>true, :e=>false, :f=>false, :g=>true])

# Check if this assignment satisfies the formula:
satisfiable(cnf, assignment)

# ## Tensor Network Formulation
# Define the satisfiability problem:
sat = Satisfiability(cnf)

# The objective is to maximize the number of satisfied clauses:
objectives(sat)

# Convert to tensor network representation:
problem = GenericTensorNetwork(sat)

# ## Mathematical Background
# For each boolean variable x, we assign a degree of freedom $s_x ∈ \{0,1\}$, where:
# * 0 represents the value 'false'
# * 1 represents the value 'true'
#
# Each clause is mapped to a tensor. For example, the clause `¬x ∨ y ∨ ¬z`
# becomes a tensor labeled by $(s_x, s_y, s_z)$:
#
# ```math
# C_{s_x,s_y,s_z} = \begin{cases}
#     0 & \text{if } s_x = 1, s_y = 0, s_z = 1 \\
#     1 & \text{otherwise}
# \end{cases}
# ```
#
# Only the configuration $(s_x, s_y, s_z) = (1, 0, 1)$ makes this clause unsatisfied.

# ## Solution Analysis
# ### 1. Satisfiability Check
# Find the maximum number of clauses that can be satisfied:
num_satisfiable = solve(problem, SizeMax())[]

# ### 2. Counting Solutions
# The graph polynomial counts assignments by number of satisfied clauses:
num_satisfiable_count = read_size_count(solve(problem, GraphPolynomial())[])

# ### 3. Finding a Satisfying Assignment
# Find one satisfying assignment:
single_config = read_config(solve(problem, SingleConfigMax())[])

# Convert the bit vector to a variable assignment and verify:
solution_assignment = Dict(zip(ProblemReductions.symbols(problem.problem), single_config))
satisfiable(cnf, solution_assignment)

# ## More APIs
# The [Independent Set Problem](@ref) chapter has more examples on how to use the APIs.
