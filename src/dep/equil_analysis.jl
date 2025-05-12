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
    return Stat(Share(), dot(distr[ind_mpc], mpc) / sum(distr[ind_mpc]), desc)
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
                desc)
end



#===========================================================================
    GINI COEFFICIENT
===========================================================================#

function Gini(
    ys::Vector{<:Real}, distr::Vector{<:Real};
    desc::String="Gini coefficient"
)
    @assert size(ys)==size(distr)
    iys = sortperm(ys)

    ys_Gini = similar(ys)
    ys_Gini .= ys[iys]
    distr_Gini = similar(distr)
    distr_Gini .= distr[iys]
    Ss = [0.0; cumsum(ys_Gini.*distr_Gini)]
    return Stat(Share(), 1.0 - dot(distr, (Ss[1:end-1].+Ss[2:end]))/Ss[end], desc)
end



#===========================================================================
    QUANTILES
===========================================================================#

# Matrix to help with the computation of quantiles
function quantile_matrix(divs::Vector{<:Real}, var::Vector{<:Real}, distr::Vector{<:Real})::SparseMatrixCSC
    nq = size(divs)[1]
    # Rank nodes from lower to higher values
    iSort = sortperm(var)
    # Cumulative distribution
    sorted_distr = distr[iSort]
    cum_distr = cumsum(sorted_distr) / sum(sorted_distr)
    # Find frontiers and weights
    ind_L, ind_U, wgt = get_weights(Extrapolate(), divs, cum_distr)
    # Matrix with indicators of quantiles
        # First quantile (it's special because it has no lower bound)
        n = ind_U[1]
        rows = ones(Int, n)
        cols = 1:n |> collect
        vals = ones(Float64, n)
        vals[end] = 1-wgt[1]
        # Rest of quantiles
        for qq=2:nq
            n = ind_U[qq]-ind_L[qq-1]
            append!(rows, fill(qq, n))
            append!(cols, ind_U[qq-1]:ind_U[qq])
            append!(vals, [wgt[qq-1]; ones(Float64, n-2); 1-wgt[qq]])
        end
        # Last quantile (it's special because it has no upper bound)
        vals[end] = 1.0
    # Recover the original order
    return sparse(rows, iSort[cols], vals, nq, size(var)[1])
end
# Method to get nq equally-sized quantiles:
function quantile_matrix(nq::Int, args...)::SparseMatrixCSC
    divs = range(0,1;length=nq+1)[2:end] |> collect
    return quantile_matrix(divs, args...)
end

# Default labels
function default_labels(quantmat::AbstractArray, distr::Vector{<:Real})
    # Get the quantiles
    q = [0; round.(Int, 100*cumsum(quantmat*distr))]
    # Get the labels
    return ["P_$(q[i-1])-$(q[i])" for i in ((1:size(quantmat)[1]) .+ 1)]
end

# Computing quantiles
function get_quants(
    quantmat::AbstractArray, var::Vector{<:Real}, distr::Vector{<:Real};
    labels::Vector{<:String}=default_labels(quantmat,distr),
    desc::String="Share of total by quantile"
)
    return StatDistr(Percentage(),
                     quantmat*(var .* distr) / dot(var,distr),
                     labels, desc)
end
function get_quants(arg_divs, var::Vector{<:Real}, distr::Vector{<:Real}; kwargs...)
    return get_quants(quantile_matrix(arg_divs, var, distr), var, distr; kwargs...)
end
function get_avg_quants(
    quantmat::AbstractArray, var::Vector{<:Real}, distr::Vector{<:Real};
    labels::Vector{<:String}=default_labels(quantmat,distr),
    desc::String="Mean by quantile"
)
    return StatDistr(Mean(),
                     quantmat*(var .* distr) ./ (quantmat*distr),
                     labels, desc)
end
function get_avg_quants(arg_divs, var::Vector{<:Real}, distr::Vector{<:Real}; kwargs...)
    return get_avg_quants(quantile_matrix(arg_divs, var, distr), var, distr; kwargs...)
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
function Base.show(xs::Vector{<:Stat{<:StatisticType}})::Nothing
    for stat in xs
        println("- ", string(stat))
    end
    return nothing
end
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



#===========================================================================
    SUMMARISE RESULTS
===========================================================================#

function ss_summarise(eco::Economía, her::Herramientas)::Vector
    agg = Aggregates(eco)
    @unpack K, C, Y = agg
    @unpack hh, distr = eco
    return [
        Stat(Percentage(), eco.pr.r, "Real interest rate"),
        Stat(Share(), K/Y, "Capital to GDP"),
        Stat(Share(), C/Y, "Consumption to GDP"),
        get_average_mpc(eco, her),
        Gini(hh.S.a, distr; desc="Wealth Gini"),
        get_pct_borrowing_constrained(distr, her),
    ]
end
function ss_distributional_analysis(eco::Economía; nq::Int=5)
    @unpack hh, distr, pr = eco
    @unpack a, z = hh.S
    # Quantile computation
    quantiles_inc = get_quants( nq, pr.w*z, distr;
                                labels=["Q$(i)" for i in 1:nq],
                                desc="Share of total income by quintile")
    quantiles_wth = get_quants( nq, a, distr;
                                labels=["Q$(i)" for i in 1:nq],
                                desc="Share of total wealth by quintile")
    # Print results
    return [quantiles_inc, quantiles_wth]
end



#===========================================================================
    SHOW RESULTS
===========================================================================#

function ss_analysis(eco::Economía, her::Herramientas; kwargs...)::Nothing
    println("\nSTEADY STATE ANALYSIS")
    # Summary
    println("\nSummary")
    show(ss_summarise(eco, her))
    # Distribution
    println("\nDistributional analysis: cross-section")
    show(ss_distributional_analysis(eco; kwargs...))
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

    # DISTRIBUTIONS
    nq = 5
    quantmat_wth = quantile_matrix(nq, a, distr)
    quantmat_inc = quantile_matrix(nq, w*z, distr)

    return nothing
end