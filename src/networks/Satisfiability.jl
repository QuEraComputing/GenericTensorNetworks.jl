"""
$TYPEDEF

The [satisfiability](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Satisfiability/) problem.

Positional arguments
-------------------------------
* `cnf` is a conjunctive normal form ([`CNF`](@ref)) for specifying the satisfiability problems.
* `get_weights` are associated with clauses.

Examples
-------------------------------
```jldoctest; setup=:(using GenericTensorNetworks)
julia> @bools x y z a b c

julia> c1 = x ∨ ¬y
x ∨ ¬y

julia> c2 = c ∨ (¬a ∨ b)
c ∨ ¬a ∨ b

julia> c3 = (z ∨ ¬a) ∨ y
z ∨ ¬a ∨ y

julia> c4 = (c ∨ z) ∨ ¬b
c ∨ z ∨ ¬b

julia> cnf = (c1 ∧ c4) ∧ (c2 ∧ c3)
(x ∨ ¬y) ∧ (c ∨ z ∨ ¬b) ∧ (c ∨ ¬a ∨ b) ∧ (z ∨ ¬a ∨ y)

julia> gp = GenericTensorNetwork(Satisfiability(cnf));

julia> solve(gp, SizeMax())[]
4.0ₜ
```
""" 
energy_terms(gp::Satisfiability) = [[getfield.(c.vars, :name)...] for c in gp.cnf.clauses]
energy_tensors(x::T, c::Satisfiability) where T = [tensor_for_clause(c.cnf.clauses[i], _pow.(Ref(x), get_weights(c, i))...) for i=1:length(c.cnf.clauses)]
extra_terms(::Satisfiability{T}) where T = Vector{T}[]
extra_tensors(::Type{T}, c::Satisfiability) where T = Array{T}[]
labels(gp::Satisfiability) = unique!(vcat(energy_terms(gp)...))

# get_weights interface
get_weights(c::Satisfiability) = c.weights
get_weights(s::Satisfiability, i::Int) = [0, s.weights[i]]

function tensor_for_clause(c::CNFClause{T}, a, b) where T
    n = length(c.vars)
    map(ci->any(i->~c.vars[i].neg == ci[i], 1:n) ? b : a, Iterators.product([[0, 1] for i=1:n]...))
end
