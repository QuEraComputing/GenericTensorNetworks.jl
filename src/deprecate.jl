@deprecate Independence(args...; kwargs...) IndependentSet(args...; kwargs...)
@deprecate MaximalIndependence(args...; kwargs...) MaximalIS(args...; kwargs...)
@deprecate UnitWeight() error("UnitWeight() is deprecated. Please Use UnitWeight(::Int) instead.")
@deprecate ZeroWeight() error("ZeroWeight() is deprecated. Please Use ZeroWeight(::Int) instead.")
@deprecate HyperSpinGlass(args...; kwargs...) SpinGlass(args...; kwargs...)

@deprecate nflavor(args...; kwargs...) num_flavors(args...; kwargs...)
@deprecate labels(args...; kwargs...) variables(args...; kwargs...)
@deprecate get_weights(args...; kwargs...) weights(args...; kwargs...)
@deprecate chweights(args...; kwargs...) set_weights(args...; kwargs...)

@deprecate spinglass_energy(args...; kwargs...) energy(args...; kwargs...)
@deprecate unit_disk_graph(args...; kwargs...) UnitDiskGraph(args...; kwargs...)
@deprecate solve(problem::ConstraintSatisfactionProblem, args...; kwargs...) solve(GenericTensorNetwork(problem), args...; kwargs...)

"""
const GraphProblem = ConstraintSatisfactionProblem

Deprecated. Use `ConstraintSatisfactionProblem` instead.
"""
const GraphProblem = ConstraintSatisfactionProblem
