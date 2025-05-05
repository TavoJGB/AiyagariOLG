module Aiyagari
    # Settings
    BASE_FOLDER = dirname(@__DIR__)

    # Packages
    using IterativeSolvers  # powm!
    using LinearAlgebra     # dot()
    using Parameters        # unpack
    # using QuantEcon         # rouwenhorst(), tauchen()
    using SparseArrays      # SparseMatrixCSC

    # Load dependencies
    include("./dep/structs.jl")
        export Herramientas, Households, Firms, Economía
    include("./dep/utils.jl")
        export EGM, LinearJumps
    include("./dep/default.jl")
        export model_parameters, model_config
    include("./dep/ss_solve.jl")
        export steady
end