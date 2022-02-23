using Test
using GraphTensorNetworks

@testset "CNF" begin
    @bools x y z a b c
    println(x)
    @test x == BoolVar(:x, false)
    @test ¬x == BoolVar(:x, true)
    @test x ∨ ¬y ∨ (z ∨ (¬a ∨ b)) == CNFClause([x, ¬y, z, ¬a, b])
    c1 = x ∨ ¬y
    c2 = c ∨ (¬a ∨ b)
    c3 = (z ∨ ¬a) ∨ y
    c4 = (c ∨ z) ∨ ¬b
    println(c4)
    @test c1 ∧ c2 == CNF([c1, c2])
    @test (c1 ∧ c2) ∧ c3 == CNF([c1, c2, c3])
    @test c1 ∧ (c2 ∧ c3) == CNF([c1, c2, c3])
    @test (c1 ∧ c4) ∧ (c2 ∧ c3) == CNF([c1, c4, c2, c3])
    cnf = (c1 ∧ c4) ∧ (c2 ∧ c3)
    println(cnf)
    gp = Satisfiability(cnf)
    @test satisfiable(cnf, Dict(:x=>true, :y=>true, :z=>true, :a=>false, :b=>false, :c=>true))
    @test !satisfiable(cnf, Dict(:x=>false, :y=>true, :z=>true, :a=>false, :b=>false, :c=>true))
end

@testset "enumerating - max cut" begin
    c1 = x ∨ ¬y
    c2 = c ∨ (¬a ∨ b)
    c3 = (z ∨ ¬a) ∨ y
    c4 = (c ∨ z) ∨ ¬b
    cnf = (c1 ∧ c4) ∧ (c2 ∧ c3)
    gp = Satisfiability(cnf)

    @test solve(gp, SizeMax())[].n == 4.0
    res = best_solutions(gp; all=true)[]
    @show res
end

