"""
$TYPEDEF

Graph problem that solved by reducing to another one. Interfaces are
- [`target_problem`](@ref), get the target problem to be reduced to.
- [`extract_result`](@ref), extract the result from the returned value of the target problem solver.
"""
abstract type ReducedProblem <: GraphProblem end

"""
    target_problem(p::ReducedProblem)

Get the target problem that the source problem `p` mapped to.
"""
function target_problem end

"""
    extract_result(p::ReducedProblem, output)
    extract_result(p::ReducedProblem)

Post process the output of the target problem to get an output to the source problem.
If the output is not provided, it will return a function instead.
The result extraction rule is determined by the output type.
e.g. If the output type is `Tropical`, it will be interpreted as the energy.
If the output is a `Vector`, it will be interpreted as a configuration.
"""
function extract_result end

# the fallback, this interface is designed for GPU compatibility
function extract_result(r::ReducedProblem, res)
    return extract_result(r)(res)
end

# fixedvertices(r::ReducedProblem) = fixedvertices(target_problem(r))
# labels(r::ReducedProblem) = labels(target_problem(r))
# terms(r::ReducedProblem) = terms(target_problem(r))
# get_weights(gp::ReducedProblem, label) = get_weights(target_problem(r))
