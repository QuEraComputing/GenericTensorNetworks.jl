export case_r3, case_dc, run_task, suboptimal_counting

using Random

function case_r3(n, k=3; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    graph = (Random.seed!(seed); LightGraphs.random_regular_graph(n, k))
    @assert length(connected_components(graph)) == 1  # connected graph
    # optimize the contraction order using KaHyPar + Greedy
    optcode = idp_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40, imbalances=0:0.001:1)
    return optcode
end

function case_dc(L::Int, ρ; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    Random.seed!(seed)
    graph = diagonal_coupled_graph(rand(L, L) .< ρ)
    # optimize the contraction order using KaHyPar + Greedy, target space complexity is 2^20
    optcode = idp_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40)
    return optcode
end

function case_sq(L::Int, ρ; sc_target, seed=2)
    # generate a random regular graph of size 100, degree 3
    Random.seed!(seed)
    graph = square_lattice_graph(rand(L, L) .< ρ)
    # optimize the contraction order using KaHyPar + Greedy, target space complexity is 2^20
    optcode = idp_code(graph; method=:kahypar, sc_target=sc_target, max_group_size=40)
    return optcode
end

"""
    run_task(code, task; usecuda=false)

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
function run_task(code::NestedEinsum, task; usecuda=false)
    if task == :totalsize
        return mis_contract(1.0, code; usecuda=usecuda)
    elseif task == :maxsize
        return mis_contract(Tropical(1.0), code; usecuda=usecuda)
    elseif task == :counting
        return mis_contract(CountingTropical(1.0), code; usecuda=usecuda)
    elseif task == :idp_polynomial
        return independence_polynomial(Val(:polynomial), code; usecuda=usecuda)
    elseif task == :idp_fft
        return independence_polynomial(Val(:fft), code; usecuda=usecuda)
    elseif task == :idp_finitefield
        return independence_polynomial(Val(:finitefield), code; usecuda=usecuda)
    elseif task == :config_single
        return mis_config(code; all=false, bounding=false, usecuda=usecuda)
    elseif task == :config_single_bounded
        return mis_config(code; all=false, bounding=true, usecuda=usecuda)
    elseif task == :config_all
        return mis_config(code; all=true, bounding=false, usecuda=usecuda)
    elseif task == :config_all_bounded
        return mis_config(code; all=true, bounding=true, usecuda=usecuda)
    else
        error("unknown task $task.")
    end
end

function run_task(g::SimpleGraph, task; usecuda=false, kwargs...)
    run_task(idp_code(g; kwargs...), task; usecuda=false)
end

function suboptimal_counting(g::SimpleGraph; kwargs...)
    res0 = run_task(g, :counting)[]
    n0, c0 = res0.n, res0.c
    c1 = 0
    for v in LightGraphs.vertices(g)
        code = idp_code(g2; kwargs...)
        resv = run_task(code, :counting)[]
        @show resv.n, n0
        if resv.n == n0 - 1
            c1 += resv.c
        end
    end
    @show c0, c1
    return c0 - c1
end
