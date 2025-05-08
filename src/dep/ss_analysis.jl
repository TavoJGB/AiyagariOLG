#===========================================================================
    AUXILIARY TYPES
===========================================================================#

abstract type StatisticType end
struct Share <:StatisticType end
struct Percentage <:StatisticType end

abstract type AbstractStatistic end
struct Stat{Ts<:StatisticType} <: AbstractStatistic
    value::Float64
    desc::String
    function Stat(::Ts, value::Float64, desc::String) where {Ts<:StatisticType}
        new{Ts}(value, desc)
    end
end



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
    # Return weighted average of the MPC
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
    FORMATTING RESULTS
===========================================================================#

function fmt(x::Stat{<:StatisticType}; digits::Int=2)
    return string(round(x.value, digits=digits))
end
function fmt(x::Stat{<:Percentage}; digits::Int=2)
    return string(round(100*x.value, digits=digits)) * " %"
end
Base.string(x::AbstractStatistic) = string(x.desc) * ": \t" * fmt(x)



#===========================================================================
    SHOW RESULTS
===========================================================================#

function ss_summary(eco::Economía, her::Herramientas)::Nothing
    println("\nSTEADY STATE SUMMARY")
    for stat in _ss_summary(eco, her)
        println("- ", string(stat))
    end
    return nothing
end
function _ss_summary(eco::Economía, her::Herramientas)::Vector
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



#===========================================================================
    GRAPHS: auxiliary functions
===========================================================================#

function plot_by_group(
    xx::Vector{<:Real}, yy::Vector{<:Real}, cfg::GraphConfig, crits, args...;
    xlab::String="", ylab::String="", tit::String="",
    leglabs=repeat([""], size(crits,1))
)
    # Preliminaries
    @unpack plotsiz, fsize, leg_fsize, lwidth = cfg
    plot()
    # Main lines
    for (ii,cr) in pairs(crits)
        ind_gr = identify_group(args..., cr)
        plot!(xx[ind_gr], yy[ind_gr], label=leglabs[ii], linewidth=lwidth)
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
    a = hh.S.a
    @unpack c, a′ = hh.G
    N_z = size(her.process_z)
    malla_a = her.grid_a.nodes
    @unpack figpath=cfg

    # POLICY FUNCTIONS (by productivity group)
    # Savings
    plot_by_group(
        a, a′, cfg, [1;N_z], her, :z,
        leglabs=["low z", "high z"], tit="Policy functions: savings"
    )
    plot!(malla_a, malla_a, line=(cfg.lwidth, :dot), color=:darkgray, label="a' = a")
    Plots.savefig(figpath * "ss_apol.png")
    # Consumption
    plot_by_group(
        a, c, cfg, [1;N_z], her, :z,
        leglabs=["low z", "high z"], tit="Policy functions: consumption")
    Plots.savefig(figpath * "ss_cpol.png")

    # VALUE FUNCTION (by productivity group)
    v = get_value(hh, Q)
    plot_by_group(
        a, v, cfg, [1;N_z], her, :z,
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

    return nothing
end