module AiyagariOLG
    # Settings
    BASE_FOLDER = dirname(@__DIR__)

    # Packages
    using CSV               # to read and save parameters in CSV format
    using IterativeSolvers  # powm!
    using LinearAlgebra     # dot()
    using Parameters        # unpack
    using Plots
    using PrettyTables      # pretty_table() for display of results
    # using QuantEcon         # rouwenhorst(), tauchen()
    using SparseArrays      # SparseMatrixCSC

    # Load dependencies
    include("./dep/structs.jl")
        export Households, Firms, Economía, Aggregates
    include("./dep/utils_computation.jl")
        export EGM, LinearJumps
    include("./dep/utils_analysis.jl")
        # export
    include("./dep/input.jl")
        export build_model
    include("./dep/output.jl")
        export write_parameters, compare_results
    include("./dep/ss_solve.jl")
        export steady
    include("./dep/ss_analysis.jl")
        export ss_analysis, ss_summarise, ss_distributional_analysis, ss_mobility, ss_graphs
end