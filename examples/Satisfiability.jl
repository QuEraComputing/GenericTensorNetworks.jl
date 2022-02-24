# # Satisfiability problem

# !!! note
#     This tutorial only covers the maximal independent set problem specific features,
#     It is recommended to read the [Independent set problem](@ref) tutorial too to know more about
#     * how to optimize the tensor network contraction order,
#     * what are the other graph properties computable,
#     * how to select correct method to compute graph properties,
#     * how to compute weighted graphs and handle open vertices.

# ## Problem definition
# One can specify a satisfiable problem in the [conjuctive normal form](https://en.wikipedia.org/wiki/Conjunctive_normal_form).
# In boolean logic, a formula is in conjunctive normal form (CNF) if it is a conjunction (∧) of one or more clauses,
# where a clause is a disjunction (∨) of literals.
using GraphTensorNetworks

@bools a b c d e f g

cnf = ∧(∨(a, b, ¬d, ¬e), ∨(¬a, d, e, ¬f), ∨(f, g), ∨(¬b, c))

# To goal is to find an assignment to satisfy the above CNF.
# For a satisfiability problem at this size, we can find the following assignment to satisfy this assignment manually.
assignment = Dict([:a=>true, :b=>false, :c=>false, :d=>true, :e=>false, :f=>false, :g=>true])

satisfiable(cnf, assignment)

# We can contruct a [`Satisfiability`](@ref) problem to solve the above problem more cleverly.

problem = Satisfiability(cnf);

# ## Satisfiability and its counting
# The size of a satisfiability problem is defined by the number of satisfiable clauses.
num_satisfiable = solve(problem, SizeMax())[]

# The [`GraphPolynomial`](@ref) of a satisfiability problem counts the number of solutions that `k` clauses satisfied.
num_satisfiable_count = solve(problem, GraphPolynomial())[]

# ## Find one of the solutions
single_config = solve(problem, SingleConfigMax())[].c.data

# One will see a bit vector printed.
# One can create an assignment and check the validity with the following statement:
satisfiable(cnf, Dict(zip(labels(problem), single_config)))