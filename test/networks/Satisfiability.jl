using Test
using GenericTensorNetworks

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
    @test get_weights(gp) == NoWeight()
    @test get_weights(chweights(gp, fill(3, 4))) == fill(3,4)
    @test_throws AssertionError Satisfiability(cnf; weights=fill(3, 9))
end

@testset "enumeration - sat" begin
    @bools x y z a b c
    c1 = x ∨ ¬y
    c2 = c ∨ (¬a ∨ b)
    c3 = (z ∨ ¬a) ∨ y
    c4 = (c ∨ z) ∨ ¬b
    cnf = (c1 ∧ c4) ∧ (c2 ∧ c3)
    gp = Satisfiability(cnf)

    @test solve(gp, SizeMax())[].n == 4.0
    res = GenericTensorNetworks.best_solutions(gp; all=true)[].c.data
    for i=0:1<<6-1
        v = StaticBitVector(Bool[i>>(k-1) & 1 for k=1:6])
        if v ∈ res
            @test satisfiable(gp.cnf, Dict(zip(labels(gp), v)))
        else
            @test !satisfiable(gp.cnf, Dict(zip(labels(gp), v)))
        end
    end
end

@testset "weighted cnf" begin
    @bools x y z a b c
    c1 = x ∨ ¬y
    c2 = c ∨ (¬a ∨ b)
    c3 = (z ∨ ¬a) ∨ y
    c4 = (c ∨ z) ∨ ¬b
    cnf = (c1 ∧ c4) ∧ (c2 ∧ c3)
    gp = Satisfiability(cnf; weights=fill(2, length(cnf)))

    @test solve(gp, SizeMax())[].n == 8.0
    res = GenericTensorNetworks.best_solutions(gp; all=true)[].c.data
    for i=0:1<<6-1
        v = StaticBitVector(Bool[i>>(k-1) & 1 for k=1:6])
        if v ∈ res
            @test satisfiable(gp.cnf, Dict(zip(labels(gp), v)))
        else
            @test !satisfiable(gp.cnf, Dict(zip(labels(gp), v)))
        end
    end
end

