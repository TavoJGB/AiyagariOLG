#===========================================================================
    MARGINAL PROPENSITIES
===========================================================================#

function get_mpc(eco::Economía, her::Herramientas)
    @unpack hh, distr = eco
    @unpack S, G = hh
    @unpack process_z, states, ind = her
    # Initialise MPC vector
    mpc = Float64[]
    # For each combination of states (other than assets), compute MPCs
    for indZ in eachcol(states[:,ind.z].==(1:size(process_z))')
        append!(mpc, _get_mpc(G.c[indZ], S.a[indZ]))
    end
    return mpc
end
function _get_mpc(c::Vector{<:Real}, a::Vector{<:Real})
    return diff(c) ./ diff(a)
end
function get_average_mpc(
    eco::Economía, her::Herramientas;
    desc::String="Average MPC"
)
    @unpack distr = eco
    mpc = get_mpc(eco, her)
    # We cannot compute MPC for the richest agent of each combination
    # of states
    ind_mpc = .!identify_group(her, :a, size(her.grid_a))
    # Return weighted average of the MPC (as a share because it's between 0 and 1)
    return Stat(Share(), dot(distr[ind_mpc], mpc) / sum(distr[ind_mpc]), :c, desc)
end



#===========================================================================
    BORROWING-CONSTRAINED AGENTS
===========================================================================#

function get_pct_borrowing_constrained(
    distr::Vector{<:Real}, her::Herramientas;
    desc::String="% of borrowing-constrained agents"
)
    return Stat(Percentage(), 
                sum(distr[get_borrowing_constrained(her)]) / sum(distr),
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
    for (iq, qq) in zip(2:length(qs)-1, qs[2:end-1])
        n = ind_U[iq]-ind_L[iq-1]
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
    divs::Vector{<:Real}, var::Vector{<:Real}, distr::Vector{<:Real};
    qtype::QuantileType=BasicQuantile()
)::SparseMatrixCSC
    nq = size(divs,1)+1
    # Rank nodes from lower to higher values
    iSort = sortperm(var)
    # Cumulative distribution
    sorted_distr = distr[iSort]
    cum_distr = cumsum(sorted_distr) / sum(sorted_distr)
    # Matrix with indicators of quantiles
    rows, cols, vals = quantile_vecs(qtype, divs, cum_distr)
    # Recover the original order
    return sparse(rows, iSort[cols], vals, nq, size(var,1))
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
    quantmat_kwargs::Dict{Symbol,<:Any}=Dict(),
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



#===========================================================================
    SUMMARISE RESULTS
===========================================================================#

function ss_summarise(eco::Economía, her::Herramientas)
    agg = Aggregates(eco)
    @unpack K, C, Y = agg
    @unpack hh, distr = eco
    return (;
        r = Stat(Percentage(), eco.pr.r, :r, "Real interest rate"),
        ratio_KY = Stat(Share(), K/Y, :K, "Capital to GDP"),
        ratio_CY = Stat(Share(), C/Y, :c, "Consumption to GDP"),
        mean_mpc = get_average_mpc(eco, her),
        gini_a = Gini(hh.S.a, distr, :a; desc="Assets Gini"),
        pct_bconstr = get_pct_borrowing_constrained(distr, her)
    )
end
function ss_distributional_analysis(eco::Economía; nq::Int=5, top=0.1)
    @unpack hh, distr, pr = eco
    @unpack a, z = hh.S
    # Preliminaries
    labs = [["Q$(i)" for i in 1:nq]; "T$(round(Int,100*top))"]
    divs = [range(0,1;length=nq+1)[2:end-1] |> collect,
            [1-top]]
    qtypes = [BasicQuantile(), TopQuantile()]
    # Quantile computation
    quantiles_inc = get_quants( divs, pr.w*z, distr, :incL;
                                quantmat_kwargs=Dict(:qtypes=>qtypes), labels=labs)
    quantiles_wth = get_quants( divs, a, distr, :a;
                                quantmat_kwargs=Dict(:qtypes=>qtypes), labels=labs)
    # Print results
    return [quantiles_inc, quantiles_wth]
end



#===========================================================================
    FORMATTING RESULTS
===========================================================================#

fmt(x::Stat{<:StatisticType}; digits::Int=2) = string(round(x.value, digits=digits))
function fmt(::Percentage, x::Real; digits::Int=2)
    if digits==0
        return string(round(Int, 100*x)) * " %"
    else
        return string(round(100*x, digits=digits)) * " %"
    end
end
fmt(x::Stat{<:Percentage}; digits::Int=2) = fmt(Percentage(), x.value; digits=digits)
Base.string(x::AbstractStatistic) = string(x.desc) * ": \t" * fmt(x)



#===========================================================================
    SHOW RESULTS
===========================================================================#

# Print summary
function Base.show(xs::NamedTuple{<:Any, <:Tuple{Vararg{Stat}}})::Nothing
    for stat in values(xs)
        println("- ", string(stat))
    end
    return nothing
end

# Print distributional analysis
function Base.show(x::StatDistr{Ts}) where {Ts<:StatisticType}
    println(string(x.desc) * ":")
    for (val, lab) in zip(x)
        println("\t ", lab, ": ", fmt(Ts(), val))
    end
end
function Base.show(xs::Vector{<:StatDistr{Ts}}) where {Ts<:StatisticType}
    header = [""; xs[1].labels...]
    ncol = size(xs[1])+1
    data = Matrix{String}(undef, length(xs), ncol)
    for (i,x) in pairs(xs)
        data[i,:] .= [x.desc; fmt.(Ref(Ts()), x.values; digits=0)...]
    end
    pretty_table(data; header, alignment=[:l; fill(:c, ncol-1)])
end

# Main function
function ss_analysis(eco::Economía, her::Herramientas;
                     save_results::Bool=true,   # by default, save results in file
                     filepath = BASE_FOLDER * "/Simulations/results/latest_simulation.csv",
                     kwargs...)::Nothing
    println("\nSTEADY STATE ANALYSIS")
    # Summary
    println("\nSummary")
    ss_summ = ss_summarise(eco, her)
    show(ss_summ)
    # Distribution
    println("\nDistributional analysis: cross-section")
    ss_distr_cs = ss_distributional_analysis(eco; kwargs...)
    show(ss_distr_cs)
    # Export results
    save_results && export_csv(filepath, exportable([ss_summ; ss_distr_cs]); delim='=')
    return nothing
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
    GRAPHS
===========================================================================#

function ss_graphs(eco::Economía, her::Herramientas, cfg::GraphConfig)::Nothing
    # Unpacking
    @unpack hh, Q, distr = eco
    @unpack a, z = hh.S
    @unpack c, a′ = hh.G
    w = eco.pr.w
    N_z = size(her.process_z)
    malla_a = her.grid_a.nodes
    @unpack figpath=cfg

    # POLICY FUNCTIONS (by productivity group)
    # Savings
    plot_by_group(
        a, a′, cfg, [1;N_z], her, :z;
        leglabs=["low z", "high z"], tit="Policy functions: savings"
    )
    plot!(malla_a, malla_a, line=(cfg.lwidth, :dot), color=:darkgray, label="a' = a")
    Plots.savefig(figpath * "ss_apol.png")
    # Consumption
    plot_by_group(
        a, c, cfg, [1;N_z], her, :z;
        leglabs=["low z", "high z"], tit="Policy functions: consumption")
    Plots.savefig(figpath * "ss_cpol.png")

    # VALUE FUNCTION (by productivity group)
    v = get_value(hh, Q)
    plot_by_group(
        a, v, cfg, [1;N_z], her, :z;
        leglabs=["low z", "high z"], tit="Value functions")
    Plots.savefig(figpath * "ss_value.png")

    # WEALTH DISTRIBUTION (by productivity group)
    plot_histogram_by_group(
        a, distr, cfg, [1;N_z], her, :z;
        leglabs=["low z", "high z"], tit="Asset distribution"
    )
    Plots.savefig(figpath * "ss_asset_hist.png")
    # plot_by_group(
    #     a, distr, cfg, [1;N_z], her, :z,
    #     leglabs=["low z", "high z"], tit="Asset distribution")
    # Plots.savefig(figpath * "ss_asset_distr.png")

    # EULER ERRORS (by productivity group)
    errs_eu = err_euler(eco)
    unconstr = .!get_borrowing_constrained(eco, her)
    errs_labs = repeat([""], N_z)
    errs_labs[[1,N_z]] .= ["low z", "high z"]
    plot_by_group(
        a[unconstr], errs_eu[unconstr], cfg, 1:N_z, her.states[unconstr, her.ind.z];
        ptype=scatter!, leglabs=errs_labs, tit="Euler Errors")
    Plots.savefig(figpath * "ss_euler_err.png")
    return nothing
end