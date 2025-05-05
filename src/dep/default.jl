#===========================================================================
    DEFAULT VALUES
===========================================================================#

function model_parameters(;
    # Household parameters
    tipo_pref=CRRA(),   # type of utility function
    β=0.94,             # discount factor
    γ=1.5,              # risk aversion
    # Firm parameters
    α=1/3,              # capital share
    δ=0.1,              # depreciation rate
    # Idiosyncratic shock parameters
    ρ_z=0.966,            # AR(1) coefficient
    σ_z=0.15              # standard deviation of the shock
)
    pars_h = (; tipo_pref, β, γ)        # household parameters
    pars_f = (; α, δ)                   # firm parameters
    pars_z = (; ρ_z, σ_z)               # idiosyncratic shock parameters
    return (; pars_h, pars_f, pars_z)
end

function model_config(;
    # Grid parameters
    N_z=5, N_a=100, min_a=0, max_a=1000.0, curv_a=1/4,
    # Solvers
    cfg_r=Solver(LinearJumps(), "Capital market", 1e-6, 0.9, 100),      # K-market solver
    cfg_hh=Solver(EGM(), "Household problem", 1e-6, 1000),              # household problem solver
    cfg_distr=Solver(POWM(), "Stationary distribution", 1e-16, 1000)    # stationary distribution solver
)
    pars_grids = (; N_z, N_a, min_a, max_a, curv_a) # grid parameters
    return Configuration(pars_grids, cfg_r, cfg_hh, cfg_distr)
end



#===========================================================================
    GUESS POLICY FUNCTIONS
===========================================================================#

function guess_G!(hh::Households, pr::Prices)::Nothing
    # Unpack parameters
    @unpack r, w = pr
    @unpack z, a = hh.S
    # Update policy functions
    hh.G.c = r*a + w*z
    hh.G.a′ = a
    return nothing
end