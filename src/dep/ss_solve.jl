#===========================================================================
    HOUSEHOLDS' PROBLEM
===========================================================================#

# Error in optimality conditions
function err_euler(
    c::Vector{<:Real}, pref::Preferencias, Q::AbstractMatrix, r′::Real;
    c′::Vector{<:Real}=c
)
    @unpack β, u′ = pref
    return u′.(c) - β*(1+r′) * Q' * u′.(c′)
end
function err_euler(eco::Economía)
    @unpack hh, pr, Q = eco
    @unpack pref, G = hh
    return err_euler(G.c, pref, Q, pr.r)
end
function err_budget(G::PolicyFunctions, prices::Prices, S::StateVariables)
    @unpack c, a′ = G
    @unpack r, w = prices
    @unpack a, z = S
    return (1+r)*a + w*z - c - a′
end
function err_budget(eco::Economía)
    @unpack hh, pr = eco
    @unpack G, S = hh
    return err_budget(G, pr, S)
end

# Solving the optimality conditions for one variable
function c_euler(pref::Preferencias, c′::Vector{<:Real}, Π_trans::Matrix{<:Real}, r::Real, N::Integer, N_a::Integer)
    @unpack β, u′, inv_u′ = pref
    return reshape(inv_u′.(β*(1+r)*reshape(u′.(c′),N_a,:) * Π_trans'), N)
end
function a_budget(pr::Prices, c::Vector{<:Real}, S::StateVariables)
    @unpack r, w = pr
    a′ = S.a
    # Budget constraint
    return (a′ + c - w*S.z) / (1+r)
end
function budget_constraint(outflow1::Vector{<:Real}, prices::Prices, S::StateVariables)
    @unpack r, w = prices
    @unpack a, z = S
    # If outflow1 is consumption, then return is savings
    # If outflow1 is savings, then return is consumption
    return (1+r)*a + w*z - outflow1
end

# Guessing the policy functions
function guess_G!(hh::Households, pr::Prices)::Nothing
    # Unpack parameters
    @unpack r, w = pr
    @unpack z, a = hh.S
    # Update policy functions
    hh.G.c = r*a + w*z
    hh.G.a′ = a
    return nothing
end

# EGM optimization: auxiliary functions
function EGM_savings!(gg::Generation{<:Oldest}, args...)::Nothing
    gg.G.a′ .= 0
    return nothing
end
function EGM_savings!(
    gg::Generation, pr::Prices, c′::Vector{<:Real},
    pref::Preferencias, process_z::MarkovProcess, grid_a::AbstractGrid
)::Nothing
    # Unpack
    @unpack N, S, states = gg
    malla_a = grid_a.nodes
    # Initialise policy function for savings
    a_EGM = similar(c′)
    # Implied consumption and assets
    c_imp = c_euler(pref, c′, process_z.Π, pr.r, N, size(grid_a))
    a_imp = a_budget(pr, c_imp, S)
    # Invert to get policy function for savings
    for zz=1:size(process_z)
        ind_z = (states.z .== zz)
        a_EGM[ind_z] = interpLinear(malla_a, a_imp[ind_z], malla_a)
    end
    # Policy function bounds
    @. a_EGM = clamp(a_EGM, grid_a.min, grid_a.max)
    # Update policy function
    gg.G.a′ = a_EGM
    return nothing
end
function EGM_consumption!(gg::Generation, pr::Prices)::Nothing
    gg.G.c = budget_constraint(gg.G.a′, pr, gg.S)
    return nothing
end

# EGM: one iteration
function EGM_iter!(gg::Generation, pr::Prices, args...)::Nothing
    EGM_savings!(gg, pr, args...)
    EGM_consumption!(gg, pr)
    return nothing
end

# All households
function hh_solve!(eco::Economía, cfg::Configuration)::Nothing
    @unpack hh, fm, pr = eco
    @unpack gens, pref, process_z, grid_a = hh
    @unpack cfg_hh, cfg_distr = cfg
    # Update policy functions for each generation
    aux_get_guess(gg::Generation) = gg.G.c
    solve!(cfg_hh, aux_get_guess, gens[end], EGM_iter!, pr)   # last generation
    for (ig, g) in enumerate(gens[(end-1):-1:1])  # previous generations
        c′ = gens[ig+1].G.c
        solve!(cfg_hh, aux_get_guess, g, EGM_iter!, pr, c′, pref, process_z, grid_a)
    end
    # Q-transition matrix
    Q_matrix!(hh)
    # Stationary distribution
    distribution!(eco.hh)
    return nothing
end



#===========================================================================
    VALUE FUNCTION
===========================================================================#

# Initial guess
function guess_value(c::Vector{<:Real}, pref::Preferencias)
    @unpack β, u = pref
    return u.(c)/(1.0-β)
end
function guess_value(hh::Households)
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
function value!(hh::Households)::Nothing
    @unpack gens, pref = hh
    value!(gens[end], pref)
    for (ig, g) in enumerate(gens[(end-1):-1:1])
        v′ = gens[ig+1].v
        value!(g, pref, v′)
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

function update_w!(prices::Prices, firms::Firms)::Nothing
    prices.w = get_w(prices.r, firms)
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

Q_matrix(::Generation{<:Oldest}, args...) = NaN
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
                Π_z[zz,z′]' .* Π_a′)             # transition probabilities
    end
    # Build the sparse matrix
    return sparse(indx_Q, indy_Q, vals_Q, N, N)
end
function Q_matrix!(gg::Generation, args...)::Nothing
    gg.Q .= Q_matrix(gg, args...)
    return nothing
end
function Q_matrix!(hh::Households)::Nothing
    @unpack gens, process_z, grid_a = hh
    # Compute Q-transition matrix for each generation
    Q_matrix!(gens[end])
    for (ig, g) in enumerate(gens[(end-1):-1:1])
        Q_matrix!(g, gens[ig+1].states, process_z.Π, grid_a)
    end
    return nothing    
end



#===========================================================================
    DISTRIBUTION
===========================================================================#

function distribution!(gg::Generation{<:Newby}, grid_a, process_z, N_g::Int)::Nothing
    @unpack N, states = gg
    i0 = findfirst(grid_a.nodes .>= 0)  # assume everyone starts with (almost) zero assets
    distr = zeros(N) # Initialise distribution
    distr[states.a .== i0] .= process_z.ss_dist/N_g
    gg.distr .= distr
    return nothing
end
function distribution!(gg::Generation, prev_distr::Vector{<:Real})::Nothing
    gg.distr .= gg.Q*prev_distr
    return nothing
end
function distribution!(hh::Households)::Nothing
    @unpack gens, grid_a, process_z = hh
    N_g = size(gens,1)
    # Compute distribution for the youngest generation
    distribution!(gens[1], grid_a, process_z, N_g)
    # Compute distribution for the rest of generations
    for (i_younger, g) in enumerate(gens[2:end])
        distribution!(g, gens[i_younger].distr)
    end
    return nothing
end



#===========================================================================
    GENERAL EQUILIBRIUM
===========================================================================#

function K_market!(r_0::Real, eco::Economía, cfg::Configuration)
    # Update prices
    eco.pr.r = r_0
    update_w!(eco.pr, eco.fm)
    # Update households
    hh_solve!(eco, cfg)
    # Compute aggregates
    agg = Aggregates(eco)
    @unpack A, K, L = agg
    # Implied r
    r_new = get_r(A, L, eco.fm)
    # Return error in capital market and implied r
    return abs(A-K), r_new
end



#===========================================================================
    STEADY STATE
===========================================================================#

function steady(hh::Households, fm::Firms, cfg::Configuration; r_0)
    # Initialise economy
    eco = Economía(r_0, hh, fm)
    # General equilibrium
    solve!(cfg.cfg_r, r_0, K_market!, eco, cfg)
    # Update value function
    value!(eco.hh)
    # Return the steady state economy
    return eco
end