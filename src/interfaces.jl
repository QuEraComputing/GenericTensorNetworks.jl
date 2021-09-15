export solve

"""
    solve(problem, task; usecuda=false)

* `problem` is the graph problem with tensor network information,
* `task` is string specifying the task. Using the maximum independent set problem as an example, it can be one of
    * "size max", the maximum independent set size,
    * "counting sum", total number of independent sets,
    * "counting max", the dengeneracy of maximum independent sets (MIS),
    * "counting max2", the dengeneracy of MIS and MIS-1,
    * "counting all", independence polynomial, the polynomial number approach,
    * "counting all (fft)", independence polynomial, the fourier transformation approach,
    * "counting all (finitefield)", independence polynomial, the finite field approach,
    * "config max", one of the maximum independent set,
    * "config max (bounded)", one of the maximum independent set, the bounded version,
    * "configs max", all MIS configurations,
    * "configs max2", all MIS configurations and MIS-1 configurations,
    * "configs all", all IS configurations,
    * "configs max (bounded)", all MIS configurations, the bounded approach (much faster),
"""
function solve(gp::GraphProblem, task; usecuda=false, kwargs...)
    if task == "size max"
        return contractx(gp, Tropical(1.0); usecuda=usecuda)
    elseif task == "counting sum"
        return contractx(gp, 1.0; usecuda=usecuda)
    elseif task == "counting max"
        return contractx(gp, CountingTropical(1.0); usecuda=usecuda)
    elseif task == "counting max2"
        return contractx(gp, Max2Poly(0.0, 1.0, 1.0); usecuda=usecuda)
    elseif task == "counting all"
        return graph_polynomial(gp, Val(:polynomial); usecuda=usecuda)
    elseif task == "config max"
        return solutions(gp, CountingTropical{Float64,Float64}; all=false, usecuda=usecuda)
    elseif task == "configs max"
        return solutions(gp, CountingTropical{Float64,Float64}; all=true, usecuda=usecuda)
    elseif task == "configs max2"
        return solutions(gp, Max2Poly{Float64,Float64}; all=true, usecuda=usecuda)
    elseif task == "configs all"
        return solutions(gp, Polynomial{Float64,:x}; all=true, usecuda=usecuda)
    # extra methods
    elseif task == "counting all (fft)"
        return graph_polynomial(gp, Val(:fft); usecuda=usecuda, kwargs...)
    elseif task == "counting all (finitefield)"
        return graph_polynomial(gp, Val(:finitefield); usecuda=usecuda, kwargs...)
    elseif task == "config max (bounded)"
        return best_solutions(gp; all=false, usecuda=usecuda)
    elseif task == "configs max (bounded)"
        return best_solutions(gp; all=true, usecuda=usecuda)
    else
        error("unknown task $task.")
    end
end
