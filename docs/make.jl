using Pkg
using GraphTensorNetworks
using GraphTensorNetworks: TropicalNumbers, Polynomials, Mods, OMEinsum, OMEinsumContractionOrders
using Documenter
using DocThemeIndigo
using Literate

for each in readdir(pkgdir(GraphTensorNetworks, "examples"))
    project_dir = pkgdir(GraphTensorNetworks, "examples", each)
    isdir(project_dir) || continue
    @info "building" project_dir
    input_file = pkgdir(GraphTensorNetworks, "examples", each, "main.jl")
    output_dir = pkgdir(GraphTensorNetworks, "docs", "src", "tutorials")
    @info "executing" input_file
    Literate.markdown(input_file, output_dir; name=each, execute=false)
end

indigo = DocThemeIndigo.install(GraphTensorNetworks)
DocMeta.setdocmeta!(GraphTensorNetworks, :DocTestSetup, :(using GraphTensorNetworks); recursive=true)

makedocs(;
    modules=[GraphTensorNetworks, TropicalNumbers, Polynomials, Mods, OMEinsum, OMEinsumContractionOrders],
    authors="Jinguo Liu",
    repo="https://github.com/Happy-Diode/GraphTensorNetworks.jl/blob/{commit}{path}#{line}",
    sitename="GraphTensorNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Happy-Diode.github.io/GraphTensorNetworks.jl",
        assets=String[indigo],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Independent set problem" => "tutorials/Independence.md",
            "Maximal independent set problem" => "tutorials/MaximalIndependence.md",
            "Cutting problem" => "tutorials/MaxCut.md",
            "Matching problem" => "tutorials/Coloring.md",
            "Binary paint shop problem" => "tutorials/PaintShop.md",
            "Coloring problem" => "tutorials/Coloring.md",
            "Other problems" => "tutorials/Others.md",
        ],
        "References" => "ref.md",
    ],
    doctest=false,
)

deploydocs(;
    repo="github.com/Happy-Diode/GraphTensorNetworks.jl",
)
