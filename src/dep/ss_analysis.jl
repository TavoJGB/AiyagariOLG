#===========================================================================
    SUMMARISE RESULTS
===========================================================================#

function ss_summarise(eco::Economía)
    @unpack hh, agg = eco
    @unpack K, C, Y, L = agg
    # Assemble relevant variables
    distr = assemble(hh.gens, :distr)
    a = assemble(hh.gens, :S, :a)
    # Summarise
    return (;
        r = Stat(Percentage(), eco.pr.r, :r, "Real interest rate"),
        ratio_KY = Stat(Share(), K/Y, :K, "Capital to GDP"),
        ratio_CY = Stat(Share(), C/Y, :c, "Consumption to GDP"),
        L = Stat(Total(), L, :L, "Aggregate Labour"),
        mean_mpc = get_average_mpc(hh),
        gini_a = Gini(a, distr, :a; desc="Assets Gini"),
        pct_bconstr = get_pct_borrowing_constrained(hh; distr)
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
    @unpack hh, pr = eco
    @unpack gens = hh;
    # Assemble relevant variables
    distr = assemble(gens, :distr)
    lab_inc = assemble(gens, g -> labour_income(g.S, pr.w))
    a = assemble(gens, :S, :a)
    # Preliminaries
    divs, labs, quantmat_kwargs = preliminaries_quantile_matrix(; kwargs...)
    # Quantile computation
    quantiles_inc = get_quants( divs, lab_inc, distr, :incL;
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
    @unpack hh, pr = eco;
    @unpack gens = hh;
    # Assemble relevant variables
    Qs = getproperty.(gens[1:nt], :Q)
    newby_distr = gens[1].distr
    newby_labinc = labour_income(gens[1].S, pr.w)
    fut_distr = gens[nt+1].distr
    fut_labinc = labour_income(gens[nt+1].S, pr.w)
    # Preliminaries
    labs = ["Q$(i)" for i in 1:nq]
    # Income mobility: prospects for the bottom and top labour income quintiles
    quantmat_newby_incL=quantile_matrix(nq, newby_labinc, newby_distr)
    quantmat_fut_incL=quantile_matrix(nq, fut_labinc, fut_distr)
    fut_quants_bottom_incL = future_probabilities(newby_distr.*quantmat_newby_incL[1,:] |> collect, Qs, nt, quantmat_fut_incL, :incL; labels=labs, subgroup_label="bottom 20% income")
    fut_quants_top_incL = future_probabilities(newby_distr.*quantmat_newby_incL[end,:] |> collect, Qs, nt, quantmat_fut_incL, :incL; labels=labs, subgroup_label="top 20% income")
    # # Wealth mobility: prospects for the bottom and top labour wealth quintiles
    # quantmat_nt_a=quantile_matrix(nq, a, distr)  # SS => quantile matrix is the same now and in the future
    # fut_quants_bottom_a = future_probabilities(distr.*quantmat_nt_a[1,:] |> collect, Q, nt, quantmat_nt_a, :a; labels=labs, subgroup_label="bottom 20%")
    # fut_quants_top_a = future_probabilities(distr.*quantmat_nt_a[end,:] |> collect, Q, nt, quantmat_nt_a, :a; labels=labs, subgroup_label="top 20%")
    return (;   mob_incL=[fut_quants_bottom_incL, fut_quants_top_incL])#,
                # mob_a=[fut_quants_bottom_a, fut_quants_top_a])
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
function Base.show(x::StatFutureDistr{Ts}; years_per_period=1) where {Ts<:StatisticType}
    println(x.desc * " after $(years_per_period*x.periods) years, having started from $(x.initial_group):")
    for (val, lab) in zip(x)
        println("\t ", lab, ": ", fmt(Ts(), val))
    end
end
function Base.show(xs::Vector{<:StatFutureDistr{Ts}}; years_per_period=1) where {Ts<:StatisticType}
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
    println(xs[1].desc * " after $(years_per_period*xs[1].periods) years:")
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
    ss_mob = ss_mobility(eco; nt=3, nq=5)
    show(ss_mob.mob_incL; years_per_period=eco.time_str.years_cohort)
    # show(ss_mob.mob_a)
    # Export results
    save_results && export_csv(filepath, exportable([ss_summ; ss_distr_cs]); delim='=')
    return nothing
end



#===========================================================================
    GRAPHS
===========================================================================#

function ss_graphs(eco::Economía, cfg::GraphConfig)::Nothing
    # PRELIMINARIES
    # Unpacking
    @unpack hh, pr = eco;
    @unpack process_z, gens = hh;
    @unpack w = pr;
    # Combine generations (if needed)
    @unpack figpath, combine_gens = cfg;
    red_gens = combine(gens, combine_gens);
    # Identify least and most productive groups
    N_z = size(process_z)
    crit_z = g -> [identify_group(g.states.z, 1), identify_group(g.states.z, N_z)]

    # POLICY FUNCTIONS (by productivity group)
    # Savings
    tiled_plot(plot_generation_apol_by(gens, :a, :a′; crits=crit_z, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Policy functions: savings, by age group")
    Plots.savefig(figpath * "ss_apol_byage.png")
    # Consumption
    tiled_plot(plot_generation_by(gens, :a, :c; crits=crit_z, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Policy functions: consumption, by age group")
    Plots.savefig(figpath * "ss_cpol_byage.png")

    # VALUE FUNCTION (by productivity group)
    tiled_plot(plot_generation_by(gens, :a, :v; crits=crit_z, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Value functions by age group")
    Plots.savefig(figpath * "ss_value_byage.png")

    # WEALTH DISTRIBUTION (by productivity group)
    tiled_plot(plot_generation_by(gens, :a, :distr; crits=crit_z, labs=["Min z", "Max z"], cfg.lwidth), cfg, "Asset distribution by age group")
    Plots.savefig(figpath * "ss_asset_distr_byage.png")
    mallas_a = getproperty.(getproperty.(gens,:grid_a),:nodes)
    tiled_plot(plot_generation_distr(red_gens, mallas_a, :a; cfg.lwidth), cfg, "Asset distribution by age group")
    Plots.savefig(figpath * "ss_asset_hist_byage.png")

    # EULER ERRORS (by productivity group, ignore oldest generation)
    # plot_euler_errors(hh, pr.r, cfg)
    tiled_plot(plot_generation_euler_errors(hh; cfg.lwidth), cfg, "Euler errors by age group")
    Plots.savefig(figpath * "ss_euler_err_byage.png")
    plot_euler_errors(hh, cfg)
    Plots.savefig(figpath * "ss_euler_err.png")
    return nothing
end