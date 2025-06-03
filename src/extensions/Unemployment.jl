module Unemployment



#===========================================================================
    IMPORTS
===========================================================================#

# Types
using ..AiyagariOLG: AbstractStateVariables, AbstractHouseholds, Aggregates, AbstractStateIndices
using ..AiyagariOLG: Generation, Oldest, Newby, Preferencias, MarkovProcess, Prices

# Methods
using ..AiyagariOLG: Generations, prepare_household_builder, distribution!, Q_vecs!, decision_mat
using ..AiyagariOLG: c_euler, a_budget, budget_constraint
using ..AiyagariOLG: assemble, get_node, get_N_agents, identify_group

# Graphs
using ..AiyagariOLG: plot_generation_by, plot_generation_apol_by, plot_generation_distr
using ..AiyagariOLG: plot_generation_euler_errors_by, plot_euler_errors, tiled_plot
using ..AiyagariOLG: plot_histogram_by_group

# Functions that will be overridden/extended
import ..AiyagariOLG: Q_matrix, Q_matrix!, distribution!, ss_graphs, GraphConfig, combine

# Other
using ..AiyagariOLG: @unpack, dot, sparse, interpLinear, zip_backward, zip_forward
using ..AiyagariOLG.Plots



#===========================================================================
    STATE VARIABLES
===========================================================================#

struct UnempStateIndices <: AbstractStateIndices
    emp::Vector{<:Int}
    z::Vector{<:Int}
    a::Vector{<:Int}
end
function get_StateIndices(; N_z::Ti, N_a::Ti) where {Ti<:Integer}
    return UnempStateIndices(   kron(1:2, ones(Ti, N_z*N_a)), 
                                repeat(kron(1:N_z, ones(Ti, N_a)), 2),
                                repeat(1:N_a, N_z*2))
end
export get_StateIndices
struct CombinedUnempStateIndices <: AbstractStateIndices
    age::Vector{<:Int}
    emp::Vector{<:Int}
    z::Vector{<:Int}
    a::Vector{<:Int}
end
function combine(vec_states::Vector{<:UnempStateIndices})
    ig      = vcat([fill(i, length(states)) for (i, states) in enumerate(vec_states)]...)
    iemp    = vcat(getfield.(vec_states, :emp)...)
    iz      = vcat(getfield.(vec_states, :z)...)
    ia      = vcat(getfield.(vec_states, :a)...)
    return CombinedUnempStateIndices(ig, iemp, iz, ia)
end

struct UnempStateVariables <: AbstractStateVariables
    emp::BitVector          # Employment status (1 if employed)
    ζ                       # Age-dependent productivity
    z::Vector{<:Real}       # Idiosyncratic productivity
    a::Vector{<:Real}       # Beginning-of-period assets
end

function get_StateVariables(states, ζ, grid_z, grid_a)
    # Productivity and assets
    zz = get_node.(Ref(grid_z), states.z)
    aa = get_node.(Ref(grid_a), states.a)
    # Number of agents
    N = length(zz)
    # Employment status
    emps = (states.emp .- 1) |> BitVector
    return UnempStateVariables(emps, ζ, zz, aa)
end

export get_StateVariables



#===========================================================================
    LABOUR MARKET TRANSITIONS
===========================================================================#


struct BasicMarkov <: MarkovProcess
    Π::Matrix{<:Real}           # Transition matrix
    ss_distr::Vector{<:Real}    # Steady state distribution
    function BasicMarkov(Π)
        return new(Π, ((Π')^100000)[:, 1])
    end
end

function build_labmkt_transitions(p_ue, p_ee)
    Π = [1-p_ue p_ue;    # transition prob if currently unemployed
         1-p_ee p_ee]    # transition prob if currently employed
    return BasicMarkov(Π)
end



#===========================================================================
    HOUSEHOLDS
===========================================================================#

struct UnempHouseholds <: AbstractHouseholds
    N::Int
    gens::Vector{<:Generation}
    pref::Preferencias
    process_z::MarkovProcess
    process_emp::MarkovProcess
    function UnempHouseholds(; process_emp, process_z, tipo_pref, pref_kwargs, kwargs...)
        # Preferences
        pref = Preferencias(tipo_pref; pref_kwargs...)
        # Vector of generations
        gens = Generations(; grid_z = process_z.grid, kwargs...)
        # Total number of agents
        N = get_N_agents(gens)
        # Return structure
        return new(N, gens, pref, process_z, process_emp)
    end
end

# Builder
function build_households(pars)
    basic_kwargs = prepare_household_builder(pars)
    process_emp = build_labmkt_transitions(pars.p_ee, pars.p_ue)
    # Return structure
    return UnempHouseholds(; process_emp, basic_kwargs...)
end
export build_households

# Households' income
function labour_income(S::UnempStateVariables, w::Real)
    @unpack ζ, z, emp = S
    return w*ζ*z.*emp
end
export labour_income



#===========================================================================
    AGGREGATES
===========================================================================#

function get_aggregates(hh::AbstractHouseholds, firms, pr::Prices)
    # Unpack
    @unpack gens = hh
    @unpack ratio_KL, F = firms
    # Assemble states and policy functions
    a′ = assemble(gens, :G, :a′)
    c = assemble(gens, :G, :c)
    a = assemble(gens, :S, :a)
    labsup = assemble(gens, g -> g.S.ζ * g.S.z .* g.S.emp)
    # Distribution
    distr = assemble(gens, :distr)
    # Households
    A = dot(distr, a′)
    A0 = dot(distr, a)
    C = dot(distr, c)
    L = dot(distr, labsup)
    # Firms
    K = ratio_KL(pr.r) * L
    Y = F(K, L)
    return Aggregates(A, A0, K, L, Y, C)
end
export get_aggregates



#===========================================================================
    Q-MATRIX: specific method
===========================================================================#

function Q_matrix(
    gg::Generation, states′::UnempStateIndices, Π_z::Matrix{<:Real},
    Π_emp::Matrix{<:Real}, grid_a′
)
    # Preliminaries
    @unpack a′ = gg.G
    zz = gg.states.z  # current productivity state
    emps = gg.states.emp  # current employment status
    N = size(a′,1)
    N_z = size(Π_z, 1)
    Tr = eltype(Π_z)
    Ti = typeof(N)
    # Initialise sparse matrix and auxiliary vectors
    indx_Q = Ti[]        # row index of Q_mat
    indy_Q = Ti[]        # column index of Q_mat
    vals_Q = Tr[]        # values of Q_mat
    # Auxiliary: decision matrix
    Π_a′ = decision_mat(a′, grid_a′)
    for z′=1:N_z
        for emp′=1:2
            Q_vecs!(indx_Q, indy_Q, vals_Q,                                     # vectors that will be appended
                    findall((states′.z .== z′) .& (states′.emp .== emp′)), 1:N, # rows and columns to fill
                    (Π_z[zz,z′] .* Π_emp[emps,emp′])' .* Π_a′)                  # transition probabilities
        end
    end
    # Build the sparse matrix
    return sparse(indx_Q, indy_Q, vals_Q, N, N)
end

# Override original method
function Q_matrix!(hh::UnempHouseholds)::Nothing
    @unpack gens, process_z, process_emp = hh
    # Compute Q-transition matrix for each generation
    Q_matrix!(gens[end])
    for (g, g′) in zip_backward(gens)
        Q_matrix!(g, g′.states, process_z.Π, process_emp.Π, g′.grid_a)
    end
    return nothing    
end



#===========================================================================
    DISTRIBUTION: New Comers (overrid original method)
===========================================================================#

function distribution!(
    gg::Generation{<:Newby}, ss_distr_z::Vector{<:Real}, ss_distr_emp::Vector{<:Real},
    N_g::Int
)::Nothing
    @unpack N, states, grid_a = gg
    i0 = findfirst(grid_a.nodes .>= 0)  # assume everyone starts with (almost) zero assets
    distr = zeros(N) # Initialise distribution
    for emp in 1:2
        @. distr[(states.a == i0) & (states.emp == emp)] = ss_distr_z/N_g * ss_distr_emp[emp]
    end
    gg.distr .= distr
    return nothing
end
function distribution!(hh::UnempHouseholds)::Nothing
    @unpack gens, process_z, process_emp = hh
    N_g = size(gens,1)
    # Compute distribution for the youngest generation
    distribution!(gens[1], process_z.ss_distr, process_emp.ss_distr, N_g)
    # Compute distribution for the rest of generations
    for (g, g_prev) in zip_forward(gens)
        distribution!(g, g_prev.Q, g_prev.distr)
    end
    return nothing
end



#===========================================================================
    HOUSEHOLDS' PROBLEM: main
===========================================================================#

# Saving decision
function savings!(gg::Generation{<:Oldest}, args...)::Nothing
    gg.G.a′ .= 0
    return nothing
end
function savings!(
    gg::Generation, pr::Prices, c′::Vector{<:Real}, grid_a′,
    pref::Preferencias, Π_z::Matrix{<:Real}, Π_emp::Matrix{<:Real}
)::Nothing
    # Unpack
    @unpack N, S, states = gg
    @unpack r, w = pr
    malla_a = gg.grid_a.nodes
    malla_a′ = grid_a′.nodes
    lab_inc = labour_income(S, w)
    # Initialise policy function for savings
    a_EGM = similar(c′)
    # Transition probabilities
    Π = kron(Π_emp, Π_z)  # joint transition matrix for z and employment status
    # Implied consumption and assets
    c_imp = c_euler(pref, c′, Π, r, N, size(grid_a′))
    # Invert to get policy function for savings
    for ind_z in eachcol(states.z .== (1:size(Π_z, 1))')     # iterate over idiosyncratic productivity
        for ind_emp in eachcol(S.emp .== [false true])          # iterate over employment status
            a_imp = a_budget(c_imp[ind_z .& ind_emp], malla_a′, lab_inc[ind_z .& ind_emp], r)
            a_EGM[ind_z .& ind_emp] = interpLinear(malla_a, a_imp, malla_a′)
        end
    end
    # Policy function bounds
    @. a_EGM = clamp(a_EGM, grid_a′.min, grid_a′.max)
    # Update policy function
    gg.G.a′ = a_EGM
    return nothing
end

# Consumption decision
function consumption!(gg::Generation, pr::Prices)::Nothing
    gg.G.c = budget_constraint(gg.G.a′, pr, gg.S)
    return nothing
end

# Solve the household problem for one generation
function generation_problem!(gg::Generation{<:Oldest}, pr::Prices)::Nothing
    savings!(gg)
    consumption!(gg, pr)
    return nothing
end
function generation_problem!(
    gg::Generation, pr::Prices, c′::Vector{<:Real}, grid_a′,
    pref::Preferencias, Π_z::Matrix{<:Real}, Π_emp::Matrix{<:Real}
)::Nothing
    savings!(gg, pr, c′, grid_a′, pref, Π_z, Π_emp)
    consumption!(gg, pr)
    return nothing
end

# All households
function hh_solve!(eco)::Nothing
    @unpack hh, fm, pr = eco;
    @unpack gens, pref, process_z, process_emp = hh;
    Π_z = process_z.Π
    Π_emp = process_emp.Π
    # Update policy functions for each generation
    generation_problem!(gens[end], pr)   # last generation
    for (g, g′) in zip_backward(gens)  # previous generations
        generation_problem!(g, pr, g′.G.c, g′.grid_a, pref, Π_z, Π_emp)
    end
    # Q-transition matrix
    Q_matrix!(eco.hh)
    # Stationary distribution
    distribution!(eco.hh)
    return nothing
end
export hh_solve!



#===========================================================================
    GRAPHS
===========================================================================#

function ss_graphs(hh::UnempHouseholds, pr::Prices, cfg::GraphConfig)::Nothing
    # PRELIMINARIES
    # Unpacking
    @unpack process_z, gens = hh;
    @unpack w = pr;
    # Combine generations (if needed)
    @unpack figpath, combine_gens = cfg;
    figpath = figpath * "unemployment/"
    red_gens = combine(gens, combine_gens);
    # Identify least and most productive groups
    N_z = size(process_z)
    crit_z_u = g -> [identify_group(g.states.z, 1) identify_group(g.states.z, N_z)] .& identify_group(g.states.emp, 1)
    crit_z_e = g -> [identify_group(g.states.z, 1) identify_group(g.states.z, N_z)] .& identify_group(g.states.emp, 2)

    # POLICY FUNCTIONS (by productivity group)
    # Savings
    tiled_plot(plot_generation_apol_by(gens, :a, :a′; crits=crit_z_u, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Policy functions: savings, unemployed by age group")
    Plots.savefig(figpath * "ss_apol_u_byage.png")
    tiled_plot(plot_generation_apol_by(gens, :a, :a′; crits=crit_z_e, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Policy functions: savings, workers by age group")
    Plots.savefig(figpath * "ss_apol_e_byage.png")
    # Consumption
    tiled_plot(plot_generation_by(gens, :a, :c; crits=crit_z_u, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Policy functions: consumption, unemployed by age group")
    Plots.savefig(figpath * "ss_cpol_u_byage.png")
    tiled_plot(plot_generation_by(gens, :a, :c; crits=crit_z_e, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Policy functions: consumption, workers by age group")
    Plots.savefig(figpath * "ss_cpol_e_byage.png")

    # VALUE FUNCTION (by productivity group)
    tiled_plot(plot_generation_by(gens, :a, :v; crits=crit_z_u, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Value functions: unemployed by age group")
    Plots.savefig(figpath * "ss_value_u_byage.png")
    tiled_plot(plot_generation_by(gens, :a, :v; crits=crit_z_e, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Value functions: workers by age group")
    Plots.savefig(figpath * "ss_value_e_byage.png")

    # WEALTH DISTRIBUTION (by productivity group)
    tiled_plot(plot_generation_by(gens, :a, :distr; crits=crit_z_u, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Asset distribution: unemployed by age group")
    Plots.savefig(figpath * "ss_asset_distr_u_byage.png")
    tiled_plot(plot_generation_by(gens, :a, :distr; crits=crit_z_e, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Asset distribution: workers by age group")
    Plots.savefig(figpath * "ss_asset_distr_e_byage.png")
    mallas_a = getproperty.(getproperty.(gens,:grid_a),:nodes)
    tiled_plot(plot_generation_distr(red_gens, mallas_a, :a; cfg.lwidth), cfg, "Asset distribution by age group")
    Plots.savefig(figpath * "ss_asset_hist_byage.png")
    plot_histogram_by_group(assemble(red_gens, :S, :a), assemble(red_gens, :distr), cfg, [1;2], assemble(red_gens, :states, :emp); leglabs=["unemployed", "employed"])
    Plots.savefig(figpath * "ss_asset_hist_byemp.png")

    # EULER ERRORS (by productivity group, ignore oldest generation)
    tiled_plot( plot_generation_euler_errors_by(gens, :z, N_z;
                                                cfg.lwidth, labs=["low z"; repeat([""], N_z-2); "high z"]),
                cfg, "Euler errors by age group")
    Plots.savefig(figpath * "ss_euler_err_byage_byz.png")
    tiled_plot( plot_generation_euler_errors_by(gens, :emp, 2;
                                                cfg.lwidth, labs=["unemployed", "employed"]),
                cfg, "Euler errors by age group")
    Plots.savefig(figpath * "ss_euler_err_byage_byemp.png")
    plot_euler_errors(hh, cfg)
    Plots.savefig(figpath * "ss_euler_err.png")
    return nothing
end
export ss_graphs

end