export case_r3, case_dc, solve, suboptimal_counting

using Random

mis_size(gp::Independence; usecuda=false) = sum(contractx(gp, TropicalF64(1.0); usecuda=usecuda)).n
mis_count(gp::Independence; usecuda=false) = sum(contractx(gp, CountingTropical{Float64,Float64}(1.0, 1.0); usecuda=usecuda)).c

function graph_polynomial(which, approach::Val, g::SimpleGraph; optmethod=:kahypar, sc_target=17, max_group_size=40, nrepeat=10, imbalances=0.0:0.001:0.8, kwargs...)
    gp = which(g; optmethod=optmethod, sc_target=sc_target, max_group_size=max_group_size, nrepeat=nrepeat, imbalances=imbalances)
    graph_polynomial(gp, approach; kwargs...)
end

function case_r3(n, k=3; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    graph = (Random.seed!(seed); LightGraphs.random_regular_graph(n, k))
    @assert length(connected_components(graph)) == 1  # connected graph
    # optimize the contraction order using KaHyPar + Greedy
    optcode = independence_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40, imbalances=0:0.001:1)
    return optcode
end

function case_dc(L::Int, ρ; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    Random.seed!(seed)
    graph = diagonal_coupled_graph(rand(L, L) .< ρ)
    # optimize the contraction order using KaHyPar + Greedy, target space complexity is 2^20
    optcode = independence_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40)
    return optcode
end

function case_sq(L::Int, ρ; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    Random.seed!(seed)
    graph = square_lattice_graph(rand(L, L) .< ρ)
    # optimize the contraction order using KaHyPar + Greedy, target space complexity is 2^20
    optcode = independence_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40)
    return optcode
end

"""
    solve(problem, , task; usecuda=false)

* `code` is the einsum code,
* `task` is one of
    * `:totalsize`, total number of independent sets,
    * `:maxsize`, the maximum independent set size,
    * `:counting`, the dengeneracy is MIS,
    * `:idp_polynomial`, independence polynomial, the polynomial number approach,
    * `:idp_fft`, independence polynomial, the fast fourier transformation approach,
    * `:idp_finitefield`, independence polynomial, the finite field approach,
    * `:config_single`, single MIS configuration,
    * `:config_single_bounded`, single MIS configuration, the bounded approach (maybe faster),
    * `:config_all`, all MIS configurations,
    * `:config_all_bounded`, all MIS configurations, the bounded approach (much faster),
"""
function solve(gp::GraphProblem, task; usecuda=false)
    if task == :totalsize
        return contractx(gp, 1.0; usecuda=usecuda)
    elseif task == :maxsize
        return contractx(gp, Tropical(1.0); usecuda=usecuda)
    elseif task == :counting
        return contractx(gp, CountingTropical(1.0); usecuda=usecuda)
    elseif task == :idp_polynomial
        return graph_polynomial(gp, Val(:polynomial); usecuda=usecuda)
    elseif task == :idp_fft
        return graph_polynomial(gp, Val(:fft); usecuda=usecuda)
    elseif task == :idp_finitefield
        return graph_polynomial(gp, Val(:finitefield); usecuda=usecuda)
    elseif task == :config_single
        return solutions(gp, CountingTropical{Float64}; all=false, usecuda=usecuda)
    elseif task == :config_single_bounded
        return optimalsolutions(gp; all=false, usecuda=usecuda)
    elseif task == :config_all
        return solutions(gp, CountingTropical{Float64}; all=true, usecuda=usecuda)
    elseif task == :config_all_bounded
        return optimalsolutions(gp; all=true, usecuda=usecuda)
    else
        error("unknown task $task.")
    end
end

function solve_is(g::SimpleGraph, task; usecuda=false)
    solve(optimize_code(Independence(g); method=:auto), task; usecuda=usecuda)
end