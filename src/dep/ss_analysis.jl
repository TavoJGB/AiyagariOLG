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
    println("STEADY STATE SUMMARY")
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