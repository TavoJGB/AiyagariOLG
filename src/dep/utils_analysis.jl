#===========================================================================
    IDENTIFICATION OF AGENTS
===========================================================================#

function identify_group(var::Vector{<:Real}, crit::Function)
    return crit.(var)
end
function identify_group(var::Vector{<:Real}, crit::Int)
    return identify_group(var, x -> x.==crit)
end
function identify_group(states::StateIndices, keyvar::Symbol, crit::Function)
    return crit.(getproperty(states, keyvar))
end
function identify_group(states::StateIndices, keyvar::Symbol, crit::Int)
    return identify_group(states, keyvar, x -> x.==crit)
end
function identify_group(G::PolicyFunctions, keyvar::Symbol, crit::Function)
    return crit.(getproperty(G, keyvar))
end

abstract type AbstractTiming end
struct EndOfPeriod <: AbstractTiming end
struct BeginningOfPeriod <: AbstractTiming end
get_saving_symbols(::EndOfPeriod) = (:G, :a′)
get_saving_symbols(::BeginningOfPeriod) = (:states, :a)

# Borrowing contrained agents: end-of-period assets
get_borrowing_constrained(a′, min_a) = a′ .<= min_a
function get_borrowing_constrained(::EndOfPeriod, gens::Vector{<:Generation}, min_a)
    return vcat([get_borrowing_constrained(g.G.a′, min_a) for g in gens]...)
end

function get_borrowing_constrained(::EndOfPeriod, hh::Households)
    @unpack gens, grid_a = hh
    a′ = assemble(gens, :G, :a′)
    return get_borrowing_constrained(a′, grid_a.min)
end

# Borrowing contrained agents: beggining-of-period assets
function get_borrowing_constrained(::BeginningOfPeriod, gens::Vector{<:Generation})
    return vcat([identify_group(g.states, :a, 1) for g in gens]...)
end
get_borrowing_constrained(::BeginningOfPeriod, hh::Households) = get_borrowing_constrained(BeginningOfPeriod(), hh.gens)

# By default: end-of-period assets
get_borrowing_constrained(gens::Vector{Generation}, args...) = get_borrowing_constrained(EndOfPeriod(), gens, args...)
get_borrowing_constrained(hh::Households) = get_borrowing_constrained(EndOfPeriod(), hh)



#===========================================================================
    MARGINAL PROPENSITIES
===========================================================================#

function _get_mpc(c::Vector{<:Real}, a::Vector{<:Real})
    return diff(c) ./ diff(a)
end
function get_mpc(g::Generation, N_z::Int)
    @unpack states = g
    # Unpack relevant variables
    @unpack a = g.S
    @unpack c = g.G
    # Initialise MPC vector
    mpc = Float64[]
    # For each combination of states (other than assets), compute MPCs
    for indZ in eachcol(states.z .== (1:N_z)')
        append!(mpc, [_get_mpc(c[indZ], a[indZ]); NaN])
    end
    return mpc
end
function get_average_mpc(hh::Households; desc::String="Average MPC")
    # Preliminaries
    @unpack process_z, gens = hh
    mpc = assemble(gens, get_mpc, size(process_z))
    distr = assemble(gens, :distr)
    # We cannot compute MPC for the richest agent of each combination
    # of states
    ind_mpc = @. !isnan(mpc)
    mpc = mpc[ind_mpc]
    distr = distr[ind_mpc]
    # Return weighted average of the MPC (as a share because it's between 0 and 1)
    return Stat(Share(), dot(distr, mpc) / sum(distr), :c, desc)
end



#===========================================================================
    BORROWING-CONSTRAINED AGENTS
===========================================================================#

function get_pct_borrowing_constrained(
    hh::Households;
    distr=assemble(hh.gens, :distr), desc::String="% of borrowing-constrained agents"
)
    return Stat(Percentage(),
                sum(distr[get_borrowing_constrained(BeginningOfPeriod(), hh)]) / sum(distr),
                :a, desc)
end



#===========================================================================
    GINI COEFFICIENT
===========================================================================#

function Gini(
    ys::Vector{<:Real}, distr::Vector{<:Real}, keyvar::Symbol;
    desc::String="Gini coefficient"
)
    @assert size(ys)==size(distr)
    iys = sortperm(ys)

    ys_Gini = ys[iys]
    distr_Gini = distr[iys]
    Ss = [0.0; cumsum(ys_Gini.*distr_Gini)]
    return Stat(Share(), 1.0 - dot(distr_Gini, (Ss[1:end-1].+Ss[2:end]))/Ss[end], keyvar, desc)
end



#===========================================================================
    QUANTILES
===========================================================================#

abstract type QuantileType end
struct BasicQuantile <: QuantileType end
struct TopQuantile <: QuantileType end
struct BottomQuantile <: QuantileType end

# Auxiliary function: gives the number of quantiles
get_qs(::BasicQuantile, divs; current_q::Int=1) = (0:size(divs,1)) .+ current_q
get_qs(::TopQuantile, divs; current_q::Int=1) = (0:size(divs,1)-1) .+ current_q
get_qs(::BottomQuantile, divs; current_q::Int=1) = (0:size(divs,1)-1) .+ current_q

# Function to assign individuals to quantiles and return vectors with
# indexes and values
# 1. QuantileType-specific functions
function _quantile_vecs!(
    rows::Vector{<:Int}, cols::Vector{<:Int}, vals::Vector{<:Real},
    ::BasicQuantile, ind_L::Vector{<:Int}, ind_U::Vector{<:Int}, wgt::Vector{<:Real}, N::Int;
    qs::AbstractArray  # index of each quantile
)::Nothing
    # First quantile (it's special because it has no lower bound)
    _quantile_vecs!(rows, cols, vals, BottomQuantile(), ind_L[1], ind_U[1], wgt[1], N; qs=1)
    # Middle quantiles
    for (iq, qq) in zip(2:(length(qs)-1), qs[2:(end-1)])
        n = ind_U[iq]-ind_L[iq-1]
        if n<=2 # Skip quantiles with no individuals
            @warn "Quantile $(qq) does not have enough individuals assigned. Skipping."
            continue
            #= This happens when there is a jump in the cumulative distribution,
            such that for some quantile there are no individuals assigned. 
            In that case, I get the following error:

            ERROR: ArgumentError: invalid GenericMemory size: too large for system address width
            Stacktrace:
            [1] GenericMemory
                @ .\boot.jl:516 [inlined]
            [...]
            
            =#
        end
        append!(rows, fill(qq, n))
        append!(cols, ind_U[iq-1]:ind_U[iq])
        append!(vals, [wgt[iq-1]; ones(Float64, n-2); 1-wgt[iq]])
    end
    # Last quantile (it's special because it has no upper bound)
    _quantile_vecs!(rows, cols, vals, TopQuantile(), ind_L[end], ind_U[end], wgt[end], N; qs=qs[end])
    return nothing
end
function _quantile_vecs!(
    rows::Vector{<:Int}, cols::Vector{<:Int}, vals::Vector{<:Real},
    ::TopQuantile, ::Any, ind_U, wgt, N::Int;
    qs  # index of each quantile
)::Nothing
    for (iq, qq) in pairs(qs)
        n = N-ind_U[iq]+1
        append!(rows, fill(qq, n))
        append!(cols, ind_U[iq]:N)
        append!(vals, [wgt[iq]; ones(Float64, n-1)])
    end
    return nothing
end
function _quantile_vecs!(
    rows::Vector{<:Int}, cols::Vector{<:Int}, vals::Vector{<:Real},
    ::BottomQuantile, ::Any, ind_U, wgt, ::Int;
    qs  # index of each quantile
)::Nothing
    for (iq, qq) in pairs(qs)
        n = ind_U[iq]
        append!(rows, fill(qq, n))
        append!(cols, 1:n)
        append!(vals, [ones(Float64, n-1); 1-wgt[iq]])
    end
    return nothing
end

# Function to assign individuals to quantiles and return vectors with
# indexes and values
# 1. General functions
function quantile_vecs!(
    rows::Vector{<:Int}, cols::Vector{<:Int}, vals::Vector{<:Real},
    qtype::QuantileType, divs::Vector{<:Real}, cum_distr::Vector{<:Real};
    qs=get_qs(qtype, divs)
)::Nothing
    # Find frontiers and weights
    ind_L, ind_U, wgt = get_weights(Extrapolate(), divs, cum_distr)
    # Assign quantiles
    _quantile_vecs!(rows, cols, vals, qtype, ind_L, ind_U, wgt, size(cum_distr,1); qs)
    return nothing
end
function quantile_vecs(
    qtype::QuantileType, divs::Vector{<:Real}, cum_distr::Vector{<:Real};
    kwargs...
)
    # Preliminaries
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    quantile_vecs!(rows, cols, vals, qtype, divs, cum_distr; kwargs...)
    return rows, cols, vals
end

# Matrix to help with the computation of quantiles
function quantile_matrix(
    divs::Vector{<:Real}, varvals::Vector{<:Real}, distr::Vector{<:Real};
    qtype::QuantileType=BasicQuantile()
)::SparseMatrixCSC
    nq = size(divs,1)+1
    # Rank nodes from lower to higher values
    iSort = sortperm(varvals)
    # Cumulative distribution
    sorted_distr = distr[iSort]
    cum_distr = cumsum(sorted_distr) / sum(sorted_distr)
    # Matrix with indicators of quantiles
    rows, cols, vals = quantile_vecs(qtype, divs, cum_distr)
    # Recover the original order
    return sparse(rows, iSort[cols], vals, nq, size(varvals,1))
end
function quantile_matrix(
    vecvec_divs::Vector{<:Vector{<:Real}}, var::Vector{<:Real}, distr::Vector{<:Real};
    qtypes::Vector{<:QuantileType}
)::SparseMatrixCSC
    # Preliminaries
    current_q = 1
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    # Rank nodes from lower to higher values
    iSort = sortperm(var)
    # Cumulative distribution
    sorted_distr = distr[iSort]
    cum_distr = cumsum(sorted_distr) / sum(sorted_distr)
    # Matrix with indicators of quantiles
    for (divs, qtype) in zip(vecvec_divs, qtypes)
        # Update quantile indexes
        qs=get_qs(qtype, divs; current_q)
        current_q += size(qs,1)
        # Get quantile vectors
        quantile_vecs!(rows, cols, vals, qtype, divs, cum_distr; qs)
    end
    # Recover the original order
    return sparse(rows, iSort[cols], vals)
end
# Method to get nq equally-sized quantiles:
function quantile_matrix(nq::Int, args...)::SparseMatrixCSC
    divs = range(0,1;length=nq+1)[2:end-1] |> collect
    return quantile_matrix(divs, args...)
end

# Default labels
function default_labels(quantmat::AbstractArray, distr::Vector{<:Real})
    # Get the quantiles
    q = [0; round.(Int, 100*cumsum(quantmat*distr))]
    # Get the labels
    return ["P_$(q[i-1])-$(q[i])" for i in ((1:size(quantmat,1)) .+ 1)]
end
function default_labels(quantmat::AbstractArray)
    # Get the quantiles
    q = range(0,100;length=(1+size(quantmat,1)))
    # Get the labels
    return ["P_$(q[i-1])-$(q[i])" for i in ((1:size(quantmat,1)) .+ 1)]
end

# Computing quantiles: shares
function get_quants(
    quantmat::SparseMatrixCSC, var::Vector{<:Real}, distr::Vector{<:Real}, keyvar::Symbol;
    labels::Vector{<:String}=default_labels(quantmat,distr),
    desc::String="Share of total $(get_var_string(keyvar)) by quantile"
)
    return StatDistr(Percentage(),
                     quantmat*(var .* distr) / dot(var,distr),
                     labels, keyvar, desc)
end
function get_quants(
    arg_divs, var::Vector{<:Real}, distr::Vector{<:Real}, args...;
    quantmat_kwargs::Dict=Dict(),
    kwargs...
)
    return get_quants(  quantile_matrix(arg_divs, var, distr; quantmat_kwargs...),
                        var, distr, args...; kwargs...)
end

# Computing quantiles: means
function get_avg_quants(
    quantmat::AbstractArray, var::Vector{<:Real}, distr::Vector{<:Real}, keyvar::Symbol;
    labels::Vector{<:String}=default_labels(quantmat,distr),
    desc::String="Mean $(get_var_string(keyvar)) by quantile"
)
    return StatDistr(Mean(),
                     quantmat*(var .* distr) ./ (quantmat*distr),
                     labels, keyvar, desc)
end
function get_avg_quants(
    arg_divs, var::Vector{<:Real}, distr::Vector{<:Real}, args...;
    quantmat_kwargs::Dict{Symbol,<:Any}=Dict(),
    kwargs...
)
    return get_avg_quants(  quantile_matrix(arg_divs, var, distr; quantmat_kwargs...),
                            var, distr, args...; kwargs...)
end

# Matrix with transition probabilities between quantiles in a period
function quantile_transition_matrix(quant_mat::SparseMatrixCSC, Q::SparseMatrixCSC, distr::Vector{<:Real})
    quant_distr = quant_mat' .* distr
    return quant_mat * Q * (quant_distr ./ sum(quant_distr,dims=1))
end



#===========================================================================
    GRAPHS: auxiliary functions
===========================================================================#

function plot_by_group(
    xx::Vector{<:Real}, yy::Vector{<:Real}, cfg::GraphConfig, crits, args...;
    ptype=plot!,
    xlab::String="", ylab::String="", tit::String="",
    leglabs=repeat([""], size(crits,1))
)
    # Preliminaries
    @unpack plotsiz, fsize, leg_fsize, lwidth = cfg
    plot()
    # Main lines
    for (ii,cr) in pairs(crits)
        ind_gr = identify_group(args..., cr)
        ptype(xx[ind_gr], yy[ind_gr], label=leglabs[ii], linewidth=lwidth)
    end
    # General settings
    xlabel!(xlab)
    ylabel!(ylab)
    title!(tit)
    plot!(size=plotsiz, tickfontsize=fsize, legendfontsize=leg_fsize)
end

function plot_histogram_by_group(
    xx::Vector{<:Real}, distr::Vector{<:Real}, cfg::GraphConfig, crits, args...;
    xlab::String="", ylab::String="", tit::String="",
    leglabs=repeat([""], size(crits,1))
)
    # Preliminaries
    @unpack plotsiz, fsize, leg_fsize, lwidth = cfg
    plot()
    # Main lines
    for (ii,cr) in pairs(crits)
        ind_gr = identify_group(args..., cr)
        stephist!(xx[ind_gr], weights=distr[ind_gr], label=leglabs[ii], linewidth=lwidth)
    end
    # General settings
    xlabel!(xlab)
    ylabel!(ylab)
    title!(tit)
    plot!(size=plotsiz, tickfontsize=fsize, legendfontsize=leg_fsize)
end



#===========================================================================
    MOBILITY
===========================================================================#

# Predict future distribution (method 1): over all the (future) state space
function future_distribution(
    distr_0::AbstractVector,        # Initial distribution
    Qs::Vector{<:SparseMatrixCSC},  # Q-transition matrix for each period ahead
    nt::Int                         # Number of periods ahead
)
    # Next-period prospects are just the corresponding rows in Q
    distr_F = Qs[1] * distr_0
    # Iterate for predictions further in the future
    for t in 2:nt
        distr_F .= Qs[t] * distr_F
    end
    # Return probability distribution of future prospects
    return distr_F
end
# Predict future distribution (method 2): over given (future) quantiles
# Equivalent to method 1 if quantmat_nt = sparse(I, size(Q_mat))
function future_distribution(
    subgroup::AbstractVector,       # Initial distribution / BitVector indicator of a subgroup of agents
    Qs::Vector{<:SparseMatrixCSC},  # Q-transition matrix for each period ahead
    nt::Int,                        # Number of periods ahead
    quantmat_nt::SparseMatrixCSC,   # Quantile matrix nt periods ahead;
    keyvar::Symbol;                 # Variable of interest
    labels::Vector{<:String}=default_labels(quantmat_nt,distr),
    desc::String="Future distribution of $(get_var_string(keyvar)) by quantile",
    subgroup_label::String="anywhere"
)
    return StatFutureDistr( Share(),
                            quantmat_nt * future_distribution(subgroup, Qs, nt),
                            labels, keyvar, desc, nt, subgroup_label)
end

# Compute probability of ending up in each given quantile, after nt periods
function future_probabilities(
    subgroup::AbstractVector,       # Initial distribution / BitVector indicator of a subgroup of agents
    Qs::Vector{<:SparseMatrixCSC},  # Q-transition matrix for each period ahead
    nt::Int,                        # Number of periods ahead
    quantmat_nt::SparseMatrixCSC,   # Quantile matrix nt periods ahead
    keyvar::Symbol;                 # Variable of interest
    labels::Vector{<:String}=default_labels(quantmat_nt),
    desc::String="Probability of reaching each future $(get_var_string(keyvar)) quantile (within cohort)",
    subgroup_label::String="anywhere"
)
    return StatFutureDistr( Probability(),
                            quantmat_nt * future_distribution(subgroup, Qs, nt) / sum(subgroup),
                            labels, keyvar, desc, nt, subgroup_label)
end