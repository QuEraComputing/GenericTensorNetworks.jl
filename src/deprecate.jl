@deprecate Independence(args...; kwargs...) IndependentSet(args...; kwargs...)
@deprecate MaximalIndependence(args...; kwargs...) MaximalIS(args...; kwargs...)
@deprecate UnitWeight() error("UnitWeight() is deprecated. Please Use UnitWeight(::Int) instead.")
@deprecate ZeroWeight() error("ZeroWeight() is deprecated. Please Use ZeroWeight(::Int) instead.")
@deprecate HyperSpinGlass(args...; kwargs...) SpinGlass(args...; kwargs...)

@deprecate nflavor(args...; kwargs...) num_flavors(args...; kwargs...)
@deprecate labels(args...; kwargs...) variables(args...; kwargs...)
@deprecate get_weights(args...; kwargs...) weights(args...; kwargs...)
@deprecate chweights(args...; kwargs...) set_weights(args...; kwargs...)

@deprecate GraphProblem() ConstraintSatisfactionProblem()