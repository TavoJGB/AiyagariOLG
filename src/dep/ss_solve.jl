#===========================================================================
    HOUSEHOLDS' PROBLEM: auxiliary
===========================================================================#

# Error in optimality conditions
function err_euler(
    c::Vector{<:Real}, pref::Preferencias, Q::AbstractMatrix, r′::Real;
    c′::Vector{<:Real}
)
    @unpack β, u′, inv_u′ = pref
    return c - inv_u′.( β*(1+r′) * Q' * u′.(c′) )
end
function err_euler(gens::Vector{<:Generation}, pref::Preferencias, r′::Real; complete::Bool=false)
    # Compute euler errors for all generations but last one
    errs = vcat([ err_euler(g.G.c, pref, g.Q, r′; c′=g′.G.c) for (g, g′) in zip_backward(gens) ][end:-1:1]...)
        # Need to invert the order because we start computing older generations, but the rest of the code
        # expects youngest generations to be the first ones in the vector
    if (complete)   # add NaN for oldest generation
        return vcat(errs, fill(NaN, size(gens[end])))
    else
        return errs
    end
end
function err_euler(eco::Economía; kwargs...)
    @unpack hh, pr = eco
    @unpack pref, gens = hh
    # Compute euler errors for all generations but last one
    return err_euler(gens, pref, pr.r; kwargs...)
end
function err_euler!(g::Generation{<:Oldest})::Nothing
    g.euler_errors .= NaN
    return nothing
end
function err_euler!(g::Generation, pref::Preferencias, r′::Real, c′)::Nothing
    g.euler_errors .= err_euler(g.G.c, pref, g.Q, r′; c′=c′)
    return nothing
end
function err_euler!(hh::AbstractHouseholds, r′)::Nothing
    @unpack gens, pref = hh
    # Compute euler errors for the oldest generation
    err_euler!(gens[end])
    # Compute distribution for the rest of generations
    for (g, g′) in zip_backward(gens)
        err_euler!(g, pref, r′, g′.G.c)
    end
    return nothing
end
function err_budget(G::PolicyFunctions, prices::Prices, S::AbstractStateVariables)
    @unpack c, a′ = G
    @unpack r, w = prices
    return S.a + income(S, prices) - c - a′
end
function err_budget(eco::Economía)
    @unpack hh, pr = eco
    return vcat([ err_budget(g.G, pr, g.S) for g in hh.gens ]...) # Unpack all generations
end

# Solving the optimality conditions for one variable
function c_euler(pref::Preferencias, c′::Vector{<:Real}, Π_trans::Matrix{<:Real}, r::Real, N::Integer, N_a′::Integer)
    @unpack β, u′, inv_u′ = pref
    return reshape(inv_u′.(β*(1+r)*reshape(u′.(c′),N_a′,:) * Π_trans'), N)
end
function a_budget( c::Vector{<:Real}, a′::Vector{<:Real}, lab_inc::Vector{<:Real}, r::Real)
    # Budget constraint
    return (a′ + c - lab_inc) / (1+r)
end
function budget_constraint(outflow1::Vector{<:Real}, S::AbstractStateVariables, args...)
    # If outflow1 is consumption, then return is savings
    # If outflow1 is savings, then return is consumption
    # Income includes capital income (r*a)
    return S.a + income(S, args...) - outflow1
end



#===========================================================================
    HOUSEHOLDS' PROBLEM: main
===========================================================================#

# code in Basic.jl or in the chosen extension



#===========================================================================
    VALUE FUNCTION
===========================================================================#

# Initial guess
function guess_value(c::Vector{<:Real}, pref::Preferencias)
    @unpack β, u = pref
    return u.(c)/(1.0-β)
end
function guess_value(hh::AbstractHouseholds)
    return guess_value(hh.G.c, hh.pref)
end

# Value function
function value!(gg::Generation{<:Oldest}, pref::Preferencias, args...)::Nothing
    gg.v .= pref.u.(gg.G.c)
    return nothing
end
function value!(gg::Generation, pref::Preferencias, v′::Vector{<:Real})::Nothing
    @unpack G, Q = gg
    @unpack c = gg.G
    @unpack β, u = pref
    gg.v .= u.(c) + β * Q' * (v′)
    return nothing
end
function value!(hh::AbstractHouseholds)::Nothing
    @unpack gens, pref = hh
    value!(gens[end], pref)
    for (g, g′) in zip_backward(gens)
        value!(g, pref, g′.v)
    end
    # hh.gens .= gens
    return nothing
end



#===========================================================================
    FIRMS' PROBLEM
===========================================================================#

function get_r(K::Real, L::Real, firms::Firms)
    @unpack F′k, δ = firms
    return F′k(K, L) - δ
end

function get_w(r::Real, firms::Firms)::Real
    @unpack α, δ = firms
    return (1.0 - α) * (α / (r + δ))^(α / (1.0 - α))
end

function update_w!(eco::AbstractEconomy)::Nothing
    @unpack fm, pr = eco
    pr.w = get_w(pr.r, fm)
    return nothing
end



#===========================================================================
    Q-TRANSITION MATRIX: auxiliary functions
===========================================================================#

# MATRIX INDICATING (WITH 1) THE POSITION (ROW) OF OPTIMAL DECISION FOR
# EACH COMINATION OF STATES (COLUMN)
function decision_mat(
    lower::Vector{<:Int}, upper::Vector{<:Int}, wgt::Vector{<:Real},
    N_dec::Int,      # Number of possible values that the decision can take
    N::Int           # Number of agents making the decision
)::SparseMatrixCSC
    return sparse(lower, 1:N, wgt, N_dec, N) + sparse(upper, 1:N, 1.0.-wgt, N_dec, N)
end
# Alternative function that computes weights
function decision_mat(
    pol_x::Vector{<:Real}, lower::Vector{<:Int}, malla_x::Vector{<:Real};
    metodo=Cap()
)::SparseMatrixCSC
    # Auxiliary variables
    N = size(lower,1)             # Number of states
    # Compute upper position and weights
    upper, wgt = get_weights(metodo, pol_x, malla_x, lower)
    # Call the main function
    return decision_mat(lower, upper, wgt, length(malla_x), N)
end
# Alternative function that first finds position in grid and computes weights
function decision_mat(
    pol_x::Vector{<:Real}, malla_x; metodo=Cap()
)::SparseMatrixCSC
    lower, upper, wgt = get_weights(metodo, pol_x, malla_x)
    return decision_mat(lower, upper, wgt, length(malla_x), length(pol_x))
end



#===========================================================================
    Q-TRANSITION MATRIX
===========================================================================#

# Columns: current states
# Rows: next-period states

# Create vector indexes to construct sparse matrix
function Q_vecs!(
    indx_Q::Vector{<:Int}, indy_Q::Vector{<:Int}, vals_Q::Vector{<:Real},
    rows, cols, part_Q::SparseMatrixCSC
)::Nothing
    # Get indexes and values
    indx,indy,vals = findnz(part_Q)
    # Update indexes
    append!(indx_Q, rows[indx])
    append!(indy_Q, cols[indy])
    append!(vals_Q, vals)
    # Return
    return nothing
end

Q_matrix(gg::Generation{<:Oldest}, args...) = sparse(I, size(gg.G.a′,1), size(gg.G.a′,1))*NaN
function Q_matrix(
    gg::Generation, states′::StateIndices, Π_z::Matrix{<:Real}, grid_a′::AbstractGrid
)
    # Preliminaries
    @unpack a′ = gg.G
    zz = gg.states.z  # current productivity state
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
        Q_vecs!(indx_Q, indy_Q, vals_Q,         # vectors that will be appended
                findall(states′.z .== z′), 1:N, # rows and columns to fill
                Π_z[zz,z′]' .* Π_a′)            # transition probabilities
    end
    # Build the sparse matrix
    return sparse(indx_Q, indy_Q, vals_Q, N, N)
end
function Q_matrix!(gg::Generation, args...)::Nothing
    gg.Q .= Q_matrix(gg, args...)
    return nothing
end
function Q_matrix!(hh::AbstractHouseholds)::Nothing
    @unpack gens, process_z = hh
    # Compute Q-transition matrix for each generation
    Q_matrix!(gens[end])
    for (g, g′) in zip_backward(gens)
        Q_matrix!(g, g′.states, process_z.Π, g′.grid_a)
    end
    return nothing    
end



#===========================================================================
    DISTRIBUTION
===========================================================================#

function distribution!(gg::Generation{<:Newby}, ss_distr_z::Vector{<:Real}, N_g::Int)::Nothing
    @unpack N, states, grid_a = gg
    i0 = findfirst(grid_a.nodes .>= 0)  # assume everyone starts with (almost) zero assets
    distr = zeros(N) # Initialise distribution
    distr[states.a .== i0] .= ss_distr_z/N_g
    gg.distr .= distr
    return nothing
end
function distribution!(gg::Generation, prev_Q::SparseMatrixCSC, prev_distr::Vector{<:Real})::Nothing
    gg.distr .= prev_Q*prev_distr
    return nothing
end
function distribution!(hh::AbstractHouseholds)::Nothing
    @unpack gens, process_z = hh
    N_g = size(gens,1)
    # Compute distribution for the youngest generation
    distribution!(gens[1], process_z.ss_distr, N_g)
    # Compute distribution for the rest of generations
    for (g, g_prev) in zip_forward(gens)
        distribution!(g, g_prev.Q, g_prev.distr)
    end
    return nothing
end



#===========================================================================
    GENERAL EQUILIBRIUM
===========================================================================#

function K_market!(r_0::Real, eco::AbstractEconomy)
    # Update prices
    eco.pr.r = r_0
    update_w!(eco)
    # Update households
    hh_solve!(eco)
    # Compute aggregates
    update_aggregates!(eco)
    @unpack A, K, L = eco.agg
    # Implied r
    r_new = get_r(A, L, eco.fm)
    # Return error in capital market and implied r
    return abs(A-K), r_new
end



#===========================================================================
    STEADY STATE
===========================================================================#

function steady(hh0::AbstractHouseholds, fm::Firms, cfg::Configuration; r_0)
    @unpack years_per_period = cfg
    # Initialise economy
    r_0 = deannualise(r_0, years_per_period)
    eco = Economía(r_0, deepcopy(hh0), fm, years_per_period);
    # General equilibrium
    solve!(cfg.cfg_r, r_0, K_market!, eco)
    # Update value function and Euler errors
    value!(eco.hh)
    err_euler!(eco.hh, eco.pr.r)    # avoid computing them after annualisation
    # Return the steady state economy
    return eco
end



#===========================================================================
    ANNUALISE RESULTS
===========================================================================#

function annualise!(eco::AbstractEconomy)::Nothing
    @unpack years_per_period = eco.time_str
    annualise!(eco.hh, years_per_period)
    annualise!(eco.pr, years_per_period)
    annualise!(eco.agg, years_per_period)
    eco.time_str.years_per_period = 1.0
    return nothing
end
function annualise!(G::PolicyFunctions, years_per_period::Real)::Nothing 
    G.c .= G.c / years_per_period
    return nothing
end
function annualise!(S::AbstractStateVariables, years_per_period::Real)::Nothing
    S.z .= S.z / years_per_period
    return nothing
end
function annualise!(hh::AbstractHouseholds, years_per_period::Real)::Nothing
    @unpack gens = hh
    for g in gens
        annualise!(g.G, years_per_period)
        annualise!(g.S, years_per_period)
    end
    return nothing
end
function annualise!(pr::Prices, years_per_period::Real)::Nothing
    pr.r = (1+pr.r)^(1/years_per_period) - 1
    return nothing
end
function deannualise(r_annual::Real, years_per_period::Real)
    return (1+r_annual)^years_per_period - 1
end
function annualise!(agg::AbstractAggregates, years_per_period::Real)::Nothing
    agg.Y /= years_per_period
    agg.C /= years_per_period
    return nothing
end