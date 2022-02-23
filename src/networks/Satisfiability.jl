struct BoolVar{T}
    name::T
    neg::Bool
end
BoolVar(name) = BoolVar(name, false)
function Base.show(io::IO, b::BoolVar)
    b.neg && print(io, "¬")
    print(io, b.name)
end

struct CNFClause{T}
    vars::Vector{BoolVar{T}}
end
function Base.show(io::IO, b::CNFClause)
    print(io, join(string.(b.vars), " ∨ "))
end
Base.:(==)(x::CNFClause, y::CNFClause) = x.vars == y.vars

struct CNF{T}
    clauses::Vector{CNFClause{T}}
end
function Base.show(io::IO, c::CNF)
    print(io, join(["($k)" for k in c.clauses], " ∧ "))
end
Base.:(==)(x::CNF, y::CNF) = x.clauses == y.clauses

¬(var::BoolVar{T}) where T = BoolVar(var.name, ~var.neg)
∨(var::BoolVar{T}, vars::BoolVar{T}...) where T = CNFClause([var, vars...])
∧(c::CNFClause{T}, cs::CNFClause{T}...) where T = CNF([c, cs...])
∨(c::CNFClause{T}, var::BoolVar{T}) where T = CNFClause([c.vars..., var])
∨(c::CNFClause{T}, d::CNFClause{T}) where T = CNFClause([c.vars..., d.vars...])
∨(var::BoolVar{T}, c::CNFClause) where T = CNFClause([var, c.vars...])
∧(c::CNFClause{T}, cs::CNF{T}) where T = CNF([c, cs.clauses...])
∧(cs::CNF{T}, c::CNFClause{T}) where T = CNF([cs.clauses..., c])
∧(cs::CNF{T}, ds::CNF{T}) where T = CNF([cs.clauses..., ds.clauses...])

macro bools(syms::Symbol...)
    esc(Expr(:block, [:($s = $BoolVar($(QuoteNode(s)))) for s in syms]..., nothing))
end

"""
    Satisfiability{CT<:AbstractEinsum,WT<:Union{UnWeighted, Vector}} <: GraphProblem
    Satisfiability(cnf::CNF; openvertices=(),
                 optimizer=GreedyMethod(), simplifier=nothing)

The [satisfiability](https://psychic-meme-f4d866f8.pages.github.io/dev/tutorials/Satisfiability.html) problem.
In the constructor, `clauses` are the conjunctive normal form (CNF) clauses for the satisfiability problems.
`optimizer` and `simplifier` are for tensor network optimization, check [`optimize_code`](@ref) for details.
"""
struct Satisfiability{CT<:AbstractEinsum,T} <: GraphProblem
    code::CT
    cnf::CNF{T}
end

function Satisfiability(cnf::CNF{T}; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing) where T
    rawcode = EinCode([[getfield.(c.vars, :name)...] for c in cnf.clauses], collect(T, openvertices))
    Satisfiability(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), cnf)
end

flavors(::Type{<:Satisfiability}) = [0, 1]  # false, true
symbols(gp::Satisfiability) = [getfield.(c.vars, :name) for c in gp.cnf.clauses]
get_weights(::Satisfiability, sym) = [0, 1]

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
function generate_tensors(fx, mi::Satisfiability{CT,T}) where {CT,T}
    cnf = mi.cnf
    ixs = getixsv(mi.code)
    isempty(cnf.clauses) && return []
	return map(1:length(cnf.clauses)) do i
        tensor_for_clause(cnf.clauses[i], fx(ixs[i])...)
    end
end

function tensor_for_clause(c::CNFClause{T}, a, b) where T
    n = length(c.vars)
    map(ci->any(i->~c.vars[i].neg == ci[i], 1:n) ? b : a, Iterators.product([[0, 1] for i=1:n]...))
end