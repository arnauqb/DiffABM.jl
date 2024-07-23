using DiffABM
using Documenter

DocMeta.setdocmeta!(DiffABM, :DocTestSetup, :(using DiffABM); recursive=true)

makedocs(;
    modules=[DiffABM],
    authors="Arnau Quera-Bofarull",
    sitename="DiffABM.jl",
    format=Documenter.HTML(;
        canonical="https://arnauqb.github.io/DiffABM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/arnauqb/DiffABM.jl",
    devbranch="main",
)
