@deprecate Independence(args...; kwargs...) IndependentSet(args...; kwargs...)
@deprecate MaximalIndependence(args...; kwargs...) MaximalIS(args...; kwargs...)
@deprecate NoWeight() UnitWeight()
@deprecate HyperSpinGlass(args...; kwargs...) SpinGlass(args...; kwargs...)

@deprecate nflavor(args...; kwargs...) num_flavors(args...; kwargs...)
@deprecate labels(args...; kwargs...) variables(args...; kwargs...)
@deprecate get_weights(args...; kwargs...) weights(args...; kwargs...)
