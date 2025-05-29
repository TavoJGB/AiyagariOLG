#===========================================================================
    SUMMARISE RESULTS
===========================================================================#

function ss_summarise(eco::Economía)
    agg = Aggregates(eco)
    @unpack K, C, Y = agg
    @unpack hh, distr = eco
    return (;
        r = Stat(Percentage(), eco.pr.r, :r, "Real interest rate"),
        ratio_KY = Stat(Share(), K/Y, :K, "Capital to GDP"),
        ratio_CY = Stat(Share(), C/Y, :c, "Consumption to GDP"),
        mean_mpc = get_average_mpc(eco),
        gini_a = Gini(hh.S.a, distr, :a; desc="Assets Gini"),
        pct_bconstr = get_pct_borrowing_constrained(distr, hh.states)
    )
end



#===========================================================================
    DISTRIBUTION AND MOBILITY
===========================================================================#

function preliminaries_quantile_matrix(; nq::Int=5, top::Real=0.0)
    labs = ["Q$(i)" for i in 1:nq]
    divs = range(0,1;length=nq+1)[2:end-1] |> collect
    if 0 < top < 1
        labs = [labs; "T$(round(Int,100*top))"]
        divs = [divs, [1-top]]
        qtypes = [BasicQuantile(), TopQuantile()]
        quantmat_kwargs=Dict(:qtypes=>qtypes)
    else
        quantmat_kwargs=Dict()
    end
    return divs, labs, quantmat_kwargs
end

function ss_distributional_analysis(eco::Economía; kwargs...)
    @unpack hh, distr, pr = eco
    @unpack a, z = hh.S
    # Preliminaries
    divs, labs, quantmat_kwargs = preliminaries_quantile_matrix(; kwargs...)
    # Quantile computation
    quantiles_inc = get_quants( divs, pr.w*z, distr, :incL;
                                quantmat_kwargs, labels=labs)
    quantiles_wth = get_quants( divs, a, distr, :a;
                                quantmat_kwargs, labels=labs)
    # Print results
    return [quantiles_inc, quantiles_wth]
end

function ss_mobility(eco::Economía;
    nt::Int,    # number of periods ahead in the future
    nq::Int=5
)
    @unpack distr, Q, hh, pr = eco
    @unpack a, z = hh.S
    # Preliminaries
    labs = ["Q$(i)" for i in 1:nq]
    # Income mobility: prospects for the bottom and top labour income quintiles
    quantmat_nt_incL=quantile_matrix(nq, pr.w*z, distr)  # SS => quantile matrix is the same now and in the future
    fut_quants_bottom_incL = future_probabilities(distr.*quantmat_nt_incL[1,:] |> collect, Q, nt, quantmat_nt_incL, :incL; labels=labs, subgroup_label="bottom 20%")
    fut_quants_top_incL = future_probabilities(distr.*quantmat_nt_incL[end,:] |> collect, Q, nt, quantmat_nt_incL, :incL; labels=labs, subgroup_label="top 20%")
    # Income mobility: prospects for the bottom and top labour income quintiles
    quantmat_nt_a=quantile_matrix(nq, a, distr)  # SS => quantile matrix is the same now and in the future
    fut_quants_bottom_a = future_probabilities(distr.*quantmat_nt_a[1,:] |> collect, Q, nt, quantmat_nt_a, :a; labels=labs, subgroup_label="bottom 20%")
    fut_quants_top_a = future_probabilities(distr.*quantmat_nt_a[end,:] |> collect, Q, nt, quantmat_nt_a, :a; labels=labs, subgroup_label="top 20%")
    return (;   mob_incL=[fut_quants_bottom_incL, fut_quants_top_incL],
                mob_a=[fut_quants_bottom_a, fut_quants_top_a])
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
fmt(::Probability, x::Real; kwargs...) = fmt(Percentage(), x; kwargs...)
fmt(x::Stat{<:Percentage}; kwargs...) = fmt(Percentage(), x.value; kwargs...)
fmt(x::Stat{<:Probability}; kwargs...) = fmt(Percentage(), x.value; kwargs...)
Base.string(x::AbstractStatistic) = string(x.desc) * ": \t" * fmt(x)



#===========================================================================
    SHOWING RESULTS
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
    println(x.desc * ":")
    for (val, lab) in zip(x)
        println("\t ", lab, ": ", fmt(Ts(), val))
    end
end
function Base.show(xs::Vector{<:StatDistr{Ts}}) where {Ts<:StatisticType}
    for x in xs[2:end]
        @assert all(x.labels .== xs[1].labels)
    end
    header = [""; xs[1].labels...]
    ncol = size(xs[1])+1
    data = Matrix{String}(undef, length(xs), ncol)
    for (i,x) in pairs(xs)
        data[i,:] .= [x.desc; fmt.(Ref(Ts()), x.values; digits=0)...]
    end
    pretty_table(data; header, alignment=[:l; fill(:c, ncol-1)])
end

# Print mobility analysis
function Base.show(x::StatFutureDistr{Ts}) where {Ts<:StatisticType}
    println(x.desc * " after $(x.periods) periods, having started from $(x.initial_group):")
    for (val, lab) in zip(x)
        println("\t ", lab, ": ", fmt(Ts(), val))
    end
end
function Base.show(xs::Vector{<:StatFutureDistr{Ts}}) where {Ts<:StatisticType}
    for x in xs[2:end]
        @assert all(x.labels .== xs[1].labels)
    end
    header = ["starting from..."; xs[1].labels...]
    ncol = size(xs[1])+1
    data = Matrix{String}(undef, length(xs), ncol)
    for (i,x) in pairs(xs)
        data[i,:] .= [x.initial_group; fmt.(Ref(Ts()), x.values; digits=0)...]
    end
    # Show
    println(xs[1].desc * " after $(xs[1].periods) periods:")
    pretty_table(data; header, alignment=[:l; fill(:c, ncol-1)])
end


# Main function
function ss_analysis(eco::Economía;
                     save_results::Bool=true,   # by default, save results in file
                     filepath = BASE_FOLDER * "/Simulations/results/latest_simulation.csv",
                     kwargs...)::Nothing
    println("\nSTEADY STATE ANALYSIS")
    # Summary
    println("\nSummary")
    ss_summ = ss_summarise(eco)
    show(ss_summ)
    # Distribution
    println("\nDistributional analysis: cross-section")
    ss_distr_cs = ss_distributional_analysis(eco; kwargs...)
    show(ss_distr_cs)
    # Distribution
    println("\nDistributional analysis: mobility")
    ss_mob = ss_mobility(eco; nt=10, nq=5)
    show(ss_mob.mob_incL)
    show(ss_mob.mob_a)
    # Export results
    save_results && export_csv(filepath, exportable([ss_summ; ss_distr_cs]); delim='=')
    return nothing
end



#===========================================================================
    GRAPHS
===========================================================================#

function ss_graphs(eco::Economía, cfg::GraphConfig)::Nothing
    # Unpacking
    @unpack hh, Q, distr = eco
    @unpack states = hh
    @unpack a, z = hh.S
    @unpack c, a′ = hh.G
    w = eco.pr.w
    N_z = size(hh.process_z)
    malla_a = hh.grid_a.nodes
    @unpack figpath=cfg

    # POLICY FUNCTIONS (by productivity group)
    # Savings
    plot_by_group(
        a, a′, cfg, [1;N_z], states, :z;
        leglabs=["low z", "high z"], tit="Policy functions: savings"
    )
    plot!(malla_a, malla_a, line=(cfg.lwidth, :dot), color=:darkgray, label="a' = a")
    Plots.savefig(figpath * "ss_apol.png")
    # Consumption
    plot_by_group(
        a, c, cfg, [1;N_z], states, :z;
        leglabs=["low z", "high z"], tit="Policy functions: consumption")
    Plots.savefig(figpath * "ss_cpol.png")

    # VALUE FUNCTION (by productivity group)
    v = get_value(hh, Q)
    plot_by_group(
        a, v, cfg, [1;N_z], states, :z;
        leglabs=["low z", "high z"], tit="Value functions")
    Plots.savefig(figpath * "ss_value.png")

    # WEALTH DISTRIBUTION (by productivity group)
    plot_histogram_by_group(
        a, distr, cfg, [1;N_z], states, :z;
        leglabs=["low z", "high z"], tit="Asset distribution"
    )
    Plots.savefig(figpath * "ss_asset_hist.png")
    # plot_by_group(
    #     a, distr, cfg, [1;N_z], :z,
    #     leglabs=["low z", "high z"], tit="Asset distribution")
    # Plots.savefig(figpath * "ss_asset_distr.png")

    # EULER ERRORS (by productivity group)
    errs_eu = err_euler(eco)
    unconstr = .!get_borrowing_constrained(eco)
    errs_labs = repeat([""], N_z)
    errs_labs[[1,N_z]] .= ["low z", "high z"]
    plot_by_group(
        a[unconstr], errs_eu[unconstr], cfg, 1:N_z, hh.states.z[unconstr];
        ptype=scatter!, leglabs=errs_labs, tit="Euler Errors")
    Plots.savefig(figpath * "ss_euler_err.png")
    return nothing
end