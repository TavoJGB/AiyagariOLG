module Basic



#===========================================================================
    IMPORTS
===========================================================================#

# Types
using ..AiyagariOLG: AbstractStateVariables, AbstractHouseholds, Aggregates, AbstractStateIndices
using ..AiyagariOLG: Generation, Oldest, Preferencias, MarkovProcess, StateIndices

# Methods
using ..AiyagariOLG: Generations, prepare_household_builder, Q_matrix!, distribution!
using ..AiyagariOLG: c_euler, a_budget, budget_constraint
using ..AiyagariOLG: assemble, get_node, get_N_agents

# Other
using ..AiyagariOLG: @unpack, dot, interpLinear, zip_backward



#===========================================================================
    STATE VARIABLES
===========================================================================#

# State indices
function get_StateIndices(; N_z::Ti, N_a::Ti) where {Ti<:Integer}
    return StateIndices(kron(1:N_z, ones(Ti, N_a)), repeat(1:N_a, N_z))
end
export get_StateIndices

# State variables
struct StateVariables <: AbstractStateVariables
    ζ                   # Age-dependent productivity
    z::Vector{<:Real}   # Idiosyncratic productivity
    a::Vector{<:Real}   # Beginning-of-period assets
end

function get_StateVariables(states, ζ, grid_z, grid_a)
    zz = get_node.(Ref(grid_z), states.z)
    aa = get_node.(Ref(grid_a), states.a)
    return StateVariables(ζ, zz, aa)
end

export get_StateVariables



#===========================================================================
    HOUSEHOLDS
===========================================================================#

struct Households <: AbstractHouseholds
    N::Int
    gens::Vector{<:Generation}
    pref::Preferencias
    process_z::MarkovProcess
    function Households(; process_z, tipo_pref, pref_kwargs, kwargs...)
        # Preferences
        pref = Preferencias(tipo_pref; pref_kwargs...)
        # Vector of generations
        gens = Generations(; grid_z = process_z.grid, kwargs...)
        # Total number of agents
        N = get_N_agents(gens)
        # Return structure
        return new(N, gens, pref, process_z)
    end
end

function build_households(pars)
    # Return structure
    return Households(; prepare_household_builder(pars)...)
end

export build_households



#===========================================================================
    AGGREGATES
===========================================================================#

function get_aggregates(hh::AbstractHouseholds, firms, prices)
    # Unpack
    @unpack gens = hh
    @unpack ratio_KL, F = firms
    # Assemble states and policy functions
    a′ = assemble(gens, :G, :a′)
    c = assemble(gens, :G, :c)
    a = assemble(gens, :S, :a)
    labsup = assemble(gens, g -> g.S.ζ * g.S.z)
    # Distribution
    distr = assemble(gens, :distr)
    # Households
    A = dot(distr, a′)
    A0 = dot(distr, a)
    C = dot(distr, c)
    L = dot(distr, labsup)
    # Firms
    K = ratio_KL(prices.r) * L
    Y = F(K, L)
    return Aggregates(A, A0, K, L, Y, C)
end
export get_aggregates



#===========================================================================
    HOUSEHOLDS' PROBLEM: main
===========================================================================#

# Households' income
function labour_income(S::AbstractStateVariables, w::Real)
    @unpack ζ, z = S
    return w*ζ*z
end
export labour_income

# Saving decision
function savings!(gg::Generation{<:Oldest}, args...)::Nothing
    gg.G.a′ .= 0
    return nothing
end
function savings!(
    gg::Generation, prices, c′::Vector{<:Real}, grid_a′,
    pref::Preferencias, Π_z::Matrix{<:Real}
)::Nothing
    # Unpack
    @unpack N, S, states = gg
    @unpack r, w = prices
    malla_a = gg.grid_a.nodes
    malla_a′ = grid_a′.nodes
    lab_inc = labour_income(S, w)
    # Initialise policy function for savings
    a_EGM = similar(c′)
    # Implied consumption and assets
    c_imp = c_euler(pref, c′, Π_z, r, N, size(grid_a′))
    # Invert to get policy function for savings
    for ind_z in eachcol(states.z .== (1:size(Π_z, 1))')
        a_imp = a_budget(c_imp[ind_z], malla_a′, lab_inc[ind_z], r)
        a_EGM[ind_z] = interpLinear(malla_a, a_imp, malla_a′)
    end
    # Policy function bounds
    @. a_EGM = clamp(a_EGM, grid_a′.min, grid_a′.max)
    # Update policy function
    gg.G.a′ = a_EGM
    return nothing
end

# Consumption decision
function consumption!(gg::Generation, prices)::Nothing
    gg.G.c = budget_constraint(gg.G.a′, prices, gg.S)
    return nothing
end

# Solve the household problem for one generation
function generation_problem!(gg::Generation{<:Oldest}, prices)::Nothing
    savings!(gg)
    consumption!(gg, prices)
    return nothing
end
function generation_problem!(
    gg::Generation, prices, c′::Vector{<:Real}, grid_a′,
    pref::Preferencias, Π_z::Matrix{<:Real}
)::Nothing
    savings!(gg, prices, c′, grid_a′, pref, Π_z)
    consumption!(gg, prices)
    return nothing
end

# All households
function hh_solve!(eco)::Nothing
    @unpack hh, fm, pr = eco;
    @unpack gens, pref, process_z = hh;
    # Update policy functions for each generation
    generation_problem!(gens[end], pr)   # last generation
    for (g, g′) in zip_backward(gens)  # previous generations
        generation_problem!(g, pr, g′.G.c, g′.grid_a, pref, process_z.Π)
    end
    # Q-transition matrix
    Q_matrix!(eco.hh)
    # Stationary distribution
    distribution!(eco.hh)
    return nothing
end
export hh_solve!



#===========================================================================
    END OF MODULE
===========================================================================#

end