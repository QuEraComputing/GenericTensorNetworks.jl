"""
    BoolVar{T}
    BoolVar(name, neg)

Boolean variable for constructing CNF clauses.
"""
struct BoolVar{T}
    name::T
    neg::Bool
end
BoolVar(name) = BoolVar(name, false)
function Base.show(io::IO, b::BoolVar)
    b.neg && print(io, "¬")
    print(io, b.name)
end

"""
    CNFClause{T}
    CNFClause(vars)

A clause in [`CNF`](@ref), its value is the logical or of `vars`, where `vars` is a vector of [`BoolVar`](@ref).
"""
struct CNFClause{T}
    vars::Vector{BoolVar{T}}
end
function Base.show(io::IO, b::CNFClause)
    print(io, join(string.(b.vars), " ∨ "))
end
Base.:(==)(x::CNFClause, y::CNFClause) = x.vars == y.vars

"""
    CNF{T}
    CNF(clauses)

Boolean expression in [conjunctive normal form](https://en.wikipedia.org/wiki/Conjunctive_normal_form).
`clauses` is a vector of [`CNFClause`](@ref), if and only if all clauses are satisfied, this CNF is satisfied.

Example
------------------------
```jldoctest; setup=:(using GenericTensorNetworks)
julia> @bools x y z

julia> cnf = (x ∨ y) ∧ (¬y ∨ z)
(x ∨ y) ∧ (¬y ∨ z)

julia> satisfiable(cnf, Dict([:x=>true, :y=>false, :z=>true]))
true

julia> satisfiable(cnf, Dict([:x=>false, :y=>false, :z=>true]))
false
```
"""
struct CNF{T}
    clauses::Vector{CNFClause{T}}
end
function Base.show(io::IO, c::CNF)
    print(io, join(["($k)" for k in c.clauses], " ∧ "))
end
Base.:(==)(x::CNF, y::CNF) = x.clauses == y.clauses
Base.length(x::CNF) = length(x.clauses)

"""
    ¬(var::BoolVar)

Negation of a boolean variables of type [`BoolVar`](@ref).
"""
¬(var::BoolVar{T}) where T = BoolVar(var.name, ~var.neg)

"""
    ∨(vars...)

Logical `or` applied on [`BoolVar`](@ref) and [`CNFClause`](@ref).
Returns a [`CNFClause`](@ref).
"""
∨(var::BoolVar{T}, vars::BoolVar{T}...) where T = CNFClause([var, vars...])
∨(c::CNFClause{T}, var::BoolVar{T}) where T = CNFClause([c.vars..., var])
∨(c::CNFClause{T}, d::CNFClause{T}) where T = CNFClause([c.vars..., d.vars...])
∨(var::BoolVar{T}, c::CNFClause) where T = CNFClause([var, c.vars...])

"""
    ∧(vars...)

Logical `and` applied on [`CNFClause`](@ref) and [`CNF`](@ref).
Returns a new [`CNF`](@ref).
"""
∧(c::CNFClause{T}, cs::CNFClause{T}...) where T = CNF([c, cs...])
∧(c::CNFClause{T}, cs::CNF{T}) where T = CNF([c, cs.clauses...])
∧(cs::CNF{T}, c::CNFClause{T}) where T = CNF([cs.clauses..., c])
∧(cs::CNF{T}, ds::CNF{T}) where T = CNF([cs.clauses..., ds.clauses...])

"""
    @bools(syms::Symbol...)

Create some boolean variables of type [`BoolVar`](@ref) in current scope that can be used in create a [`CNF`](@ref).

Example
------------------------
```jldoctest; setup=:(using GenericTensorNetworks)
julia> @bools x y z

julia> (x ∨ y) ∧ (¬y ∨ z)
(x ∨ y) ∧ (¬y ∨ z)
```
"""
macro bools(syms::Symbol...)
    esc(Expr(:block, [:($s = $BoolVar($(QuoteNode(s)))) for s in syms]..., nothing))
end

"""
$TYPEDEF

The [satisfiability](https://queracomputing.github.io/GenericTensorNetworks.jl/dev/generated/Satisfiability/) problem.

Positional arguments
-------------------------------
* `cnf` is a conjunctive normal form ([`CNF`](@ref)) for specifying the satisfiability problems.
* `weights` are associated with clauses.

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

julia> gp = Satisfiability(cnf);

julia> solve(gp, SizeMax())[]
4.0ₜ
```
"""
struct Satisfiability{T,WT<:Union{NoWeight, Vector}} <: GraphProblem
    cnf::CNF{T}
    weights::WT
    function Satisfiability(cnf::CNF{T}, weights::WT=NoWeight()) where {T}
        @assert weights isa NoWeight || length(weights) == length(cnf) "weights size inconsistent! should be $(length(cnf)), got: $(length(weights))"
        new{T, typeof(weights)}(cnf, weights)
    end
end

function GenericTensorNetwork(cfg::Satisfiability{CT,T,WT}; openvertices=(), fixedvertices=Dict{T,Int}())
    rawcode = EinCode([[getfield.(c.vars, :name)...] for c in cfg.cnf.clauses], collect(T, openvertices))
    return GenericTensorNetwork(cfg, rawcode, Dict{T,Int}(fixedvertices))
end

flavors(::Type{<:Satisfiability}) = [0, 1]  # false, true
terms(gp::Satisfiability) = getixsv(gp.code)
labels(gp::Satisfiability) = unique!(vcat(getixsv(gp.code)...))

# weights interface
get_weights(c::Satisfiability) = c.weights
get_weights(s::Satisfiability, i::Int) = [0, s.weights[i]]
chweights(c::Satisfiability, weights) = Satisfiability(c.code, c.cnf, weights, c.fixedvertices)

"""
    satisfiable(cnf::CNF, config::AbstractDict)

Returns true if an assignment of variables satisfies a [`CNF`](@ref).
"""
function satisfiable(cnf::CNF{T}, config::AbstractDict{T}) where T
    all(x->satisfiable(x, config), cnf.clauses)
end
function satisfiable(c::CNFClause{T}, config::AbstractDict{T}) where T
    any(x->satisfiable(x, config), c.vars)
end
function satisfiable(v::BoolVar{T}, config::AbstractDict{T}) where T
    config[v.name] == ~v.neg
end

# the first argument is a function of variables
function generate_tensors(x::VT, sa::Satisfiability{CT,T}) where {CT,T,VT}
    cnf = sa.cnf
    ixs = getixsv(sa.code)
    isempty(cnf.clauses) && return []
	return select_dims(
        add_labels!(
            map(1:length(cnf.clauses)) do i
                tensor_for_clause(cnf.clauses[i], _pow.(Ref(x), get_weights(sa, i))...)
            end,
            ixs, labels(sa))
        , ixs, fixedvertices(sa)
    )
end

function tensor_for_clause(c::CNFClause{T}, a, b) where T
    n = length(c.vars)
    map(ci->any(i->~c.vars[i].neg == ci[i], 1:n) ? b : a, Iterators.product([[0, 1] for i=1:n]...))
end
