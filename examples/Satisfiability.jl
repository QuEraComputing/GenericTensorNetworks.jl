# # Satisfiability problem

# !!! note
#     It is highly recommended to read the [Independent set problem](@ref) chapter before reading this one.

# ## Problem definition
# In logic and computer science, the [boolean satisfiability problem](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem) is the problem of determining if there exists an interpretation that satisfies a given boolean formula.
# One can specify a satisfiable problem in the [conjuctive normal form](https://en.wikipedia.org/wiki/Conjunctive_normal_form).
# In boolean logic, a formula is in conjunctive normal form (CNF) if it is a conjunction (∧) of one or more clauses,
# where a clause is a disjunction (∨) of literals.
using GenericTensorNetworks

@bools a b c d e f g

cnf = ∧(∨(a, b, ¬d, ¬e), ∨(¬a, d, e, ¬f), ∨(f, g), ∨(¬b, c))

# To goal is to find an assignment to satisfy the above CNF.
# For a satisfiability problem at this size, we can find the following assignment to satisfy this assignment manually.
assignment = Dict([:a=>true, :b=>false, :c=>false, :d=>true, :e=>false, :f=>false, :g=>true])

satisfiable(cnf, assignment)

# ## Generic tensor network representation
# If we contract this tensor network, we will get a multiplicative factor ``x`` whenever there is a clause satisfied.

problem = Satisfiability(cnf);

# ### Theory (can skip)
# We can contruct a [`Satisfiability`](@ref) problem to solve the above problem.
# To generate a tensor network, we map a boolean variable ``x`` and its negation ``\neg x`` to a degree of freedom (label) ``s_x \in \{0, 1\}``,
# where 0 stands for variable ``x`` having value `false` while 1 stands for having value `true`.
# Then we map a clause to a tensor. For example, a clause ``¬x ∨ y ∨ ¬z`` can be mapped to a tensor labeled by ``(s_x, s_y, s_z)``.
# ```math
# C = \left(\begin{matrix}
# \left(\begin{matrix}
# x & x \\
# x & x
# \end{matrix}\right) \\
# \left(\begin{matrix}
# x & x \\
# 1 & x
# \end{matrix}\right)
# \end{matrix}\right).
# ```
# There is only one entry ``(s_x, s_y, s_z) = (1, 0, 1)`` that makes this clause unsatisfied.

# ## Solving properties
# #### Satisfiability and its counting
# The size of a satisfiability problem is defined by the number of satisfiable clauses.
num_satisfiable = solve(problem, SizeMax())[]

# The [`GraphPolynomial`](@ref) of a satisfiability problem counts the number of solutions that `k` clauses satisfied.
num_satisfiable_count = solve(problem, GraphPolynomial())[]

# #### Find one of the solutions
single_config = solve(problem, SingleConfigMax())[].c.data

# One will see a bit vector printed.
# One can create an assignment and check the validity with the following statement:
satisfiable(cnf, Dict(zip(labels(problem), single_config)))
