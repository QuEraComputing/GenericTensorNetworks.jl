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
    gp = GenericTensorNetwork(Satisfiability(cnf))
    @test satisfiable(cnf, Dict(:x=>true, :y=>true, :z=>true, :a=>false, :b=>false, :c=>true))
    @test !satisfiable(cnf, Dict(:x=>false, :y=>true, :z=>true, :a=>false, :b=>false, :c=>true))
    @test GenericTensorNetworks.weights(gp) == UnitWeight(length(gp.problem.cnf))
    @test GenericTensorNetworks.weights(set_weights(gp, fill(3, 4))) == fill(3,4)
    @test_throws AssertionError Satisfiability(cnf, fill(3, 9))
end

@testset "enumeration - sat" begin
    @bools x y z a b c
    c1 = x ∨ ¬y
    c2 = c ∨ (¬a ∨ b)
    c3 = (z ∨ ¬a) ∨ y
    c4 = (c ∨ z) ∨ ¬b
    cnf = (c1 ∧ c4) ∧ (c2 ∧ c3)
    gp = GenericTensorNetwork(Satisfiability(cnf))

    @test solve(gp, SizeMax())[].n == 2.0
    res = GenericTensorNetworks.largest_solutions(gp; invert=true, all=true)[].c.data
    for i=0:1<<6-1
        v = StaticBitVector(Bool[i>>(k-1) & 1 for k=1:6])
        if v ∈ res
            @test satisfiable(gp.problem.cnf, Dict(zip(ProblemReductions.symbols(gp.problem), v)))
        else
            @test !satisfiable(gp.problem.cnf, Dict(zip(ProblemReductions.symbols(gp.problem), v)))
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
    gp = GenericTensorNetwork(Satisfiability(cnf, fill(2, length(cnf))))

    @test solve(gp, SizeMax())[].n == 4.0
    res = GenericTensorNetworks.largest_solutions(gp; invert=true, all=true)[].c.data
    for i=0:1<<6-1
        v = StaticBitVector(Bool[i>>(k-1) & 1 for k=1:6])
        if v ∈ res
            @test satisfiable(gp.problem.cnf, Dict(zip(ProblemReductions.symbols(gp.problem), v)))
        else
            @test !satisfiable(gp.problem.cnf, Dict(zip(ProblemReductions.symbols(gp.problem), v)))
        end
    end
end

