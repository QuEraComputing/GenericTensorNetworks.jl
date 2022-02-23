struct BoolVar{T}
    name::T
    neg::Bool
end

struct CNFClause{T}
    vars::Vector{BoolVar{T}}
end

struct CNF{T}
    clauses::Vector{CNFClause{T}}
end

Base.:∨(var::BoolVar{T}, vars::BoolVar{T}...) where T = CNFClause([var, vars...])
Base.:∧(c::CNFClause{T}, cs::CNFClause{T}...) where T = CNF([c, cs...])

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
    clauses::CNF{T}
end

function Satisfiability(cnf::CNF; openvertices=(), optimizer=GreedyMethod(), simplifier=nothing)
    rawcode = EinCode(([[c.vars...] for c in cnf.clauses]...,), collect(Int, openvertices))
    Satisfiability(_optimize_code(rawcode, uniformsize(rawcode, 2), optimizer, simplifier), weights)
end

flavors(::Type{<:Satisfiability}) = [0, 1]
symbols(gp::Satisfiability) = [i for i in 1:length(getixsv(gp.code))]
get_weights(gp::Satisfiability, label) = [0, gp.weights[findfirst(==(label), symbols(gp))]]

function generate_tensors(fx, mi::Satisfiability)
    ixs = getixsv(mi.code)
    isempty(ixs) && return []
    T = eltype(fx(ixs[1][end]))
	return map(ixs) do ix
        neighbortensor(fx(ix[end])..., length(ix))
    end
end
function neighbortensor(a::T, b::T, d::Int) where T
    t = zeros(T, fill(2, d)...)
    for i = 2:1<<(d-1)
        t[i] = one(T)
    end
    t[1<<(d-1)+1] = a
    t[1<<(d-1)+1] = b
    return t
end


"""
    is_maximal_independent_set(g::SimpleGraph, config)

Return true if `config` (a vector of boolean numbers as the mask of vertices) is a maximal independent set of graph `g`.
"""
is_maximal_independent_set(g::SimpleGraph, config) = !any(e->config[e.src] == 1 && config[e.dst] == 1, edges(g)) && all(w->config[w] == 1 || any(v->!iszero(config[v]), neighbors(g, w)), Graphs.vertices(g))