#===========================================================================
    SOLVERS
===========================================================================#

# Solvers
abstract type SolverType end
struct LinearJumps <: SolverType end
struct EGM <: SolverType end            # requires SolverEGM.jl extension

get_solver_type(solv_t::Symbol) = eval(solv_t)()

# Parameters of the solvers
abstract type SolverParameters end
@with_kw struct BasicSolverParameters <: SolverParameters
    objective::String
    tol::Real
    maxit::Int
end
SolverParameters(::SolverType; kwargs...) = BasicSolverParameters(; kwargs...)
get_required_parameters(::SolverType) = [:objective, :tol, :maxit]
@with_kw struct LinearJumpsParameters <: SolverParameters
    objective::String
    tol::Real
    wgt::Real
    maxit::Int
end
SolverParameters(::LinearJumps; kwargs...) = LinearJumpsParameters(; kwargs...)
get_required_parameters(::LinearJumps) = [:objective, :tol, :wgt, :maxit]

# Main solver structure
struct Solver{Ts <: SolverType}
    p::SolverParameters
    Solver(type::Ts; kwargs...) where {Ts <: SolverType} = new{Ts}(SolverParameters(type; kwargs...))
end



#===========================================================================
    CONFIGURATION
===========================================================================#

@with_kw struct GraphConfig
    figpath::String
    plotsiz::Vector{<:Int}
    fsize::Real
    leg_fsize::Real
    lwidth::Real
    combine_gens::Int
end
function _GraphConfig(; plotsiz::String, kwargs...)
    return GraphConfig(; plotsiz=parse.(Int, split(plotsiz,"x")), kwargs...)
end

struct Configuration
    cfg_r::Solver
    # cfg_hh::Solver
    cfg_graph::GraphConfig
    # Years per model period
    years_per_period::Int
end



#===========================================================================
    GRIDS
===========================================================================#

abstract type AbstractGridType end
struct Curved <: AbstractGridType end
struct Linear <: AbstractGridType end
struct Simple <: AbstractGridType end

abstract type AbstractGrid end

struct SimpleGrid <: AbstractGrid
    N::Int
    nodes::Vector{<:Real}
end
_SimpleGrid(; nodes) = SimpleGrid(length(nodes), nodes)
Grid(type::AbstractGridType; kwargs...) = _SimpleGrid(; kwargs...)
get_required_parameters(::AbstractGridType) = [:nodes]
struct LinearGrid <: AbstractGrid
    N::Int
    min::Real
    max::Real
    nodes::Vector{<:Real}
end
function _LinearGrid(; N::Int, min::Real, max::Real)
    nodes = range(min, max, length=N) |> collect
    return LinearGrid(N, min, max, nodes)
end
Grid(type::Linear; kwargs...) = _LinearGrid(; kwargs...)
get_required_parameters(::Linear) = [:N, :min, :max]
@with_kw struct CurvedGrid <: AbstractGrid
    N::Int
    curv::Real
    min::Real
    max::Real
    nodes::Vector{<:Real}
end
function _CurvedGrid(; N::Int, curv::Real, min::Real, max::Real)
    nodes = min .+ (max - min) .* (range(0.0, 1.0, length=N) .^ (1/curv))
    return CurvedGrid(N, curv, min, max, nodes)
end
Grid(type::Curved; kwargs...) = _CurvedGrid(; kwargs...)
get_required_parameters(::Curved) = [:N, :curv, :min, :max]

# Grid methods
Base.size(grid::AbstractGrid) = grid.N
Base.length(grid::AbstractGrid) = grid.N
function get_node(grid::AbstractGrid, i::Int)
    return grid.nodes[i]
end

# Age Dependent 
struct CombinedGrid <: AbstractGrid
    grids::Vector{<:AbstractGrid}
    CombinedGrid(grids::Vector{<:AbstractGrid}) = new(grids)
    CombinedGrid(::Curved; kwargs...) = get_AgeDependentCurvedGrid(; kwargs...)
end
function get_AgeDependentCurvedGrid(;
    max::Real, ages::AbstractVector, gscale_max::Real, age_peak::Real=sum(ages)/size(ages,1), kwargs...
)
    g_peak = findlast(ages .<= age_peak)
    N_g = size(ages,1)
    g_max_a = range(max, gscale_max*max; length=(1+maximum([g_peak-1,N_g-g_peak])))[abs.((1:N_g) .-  (g_peak)) .+ 1]
    return [_CurvedGrid(; max=max_a, kwargs...) for max_a in g_max_a]
end



#===========================================================================
    MARKOV PROCESSES
===========================================================================#

abstract type MarkovProcess end
@with_kw struct DiscreteAR1 <: MarkovProcess
    ρ::Real                     # Persistence
    σ::Real                     # Standard deviation
    grid::AbstractGrid          # Grid for the state variable
    Π::Matrix{<:Real}           # Transition matrix
    ss_distr::Vector{<:Real}    # Steady state distribution
end
function _DiscreteAR1(; N::Int, ρ::Real, σ::Real, method=rouwenhorst)
    nodes, trans = method(N, ρ, σ)
    trans = trans' |> collect
    # SS Distribution
    ss_dist = (trans^100000)[:, 1]
    # Nodes
    nodes = exp.(nodes)
    nodes .= nodes / dot(ss_dist, nodes)
    # Transition matrix
    Π = trans' |> collect
    return DiscreteAR1(ρ, σ, Grid(Simple(); nodes), Π, ss_dist)
end
get_DiscreteAR1_parameters() = [:N, :ρ, :σ]

# Methods
Base.size(mkv::MarkovProcess) = Base.size(mkv.grid)
get_node(mkv::MarkovProcess, i::Int) = get_node(mkv.grid, i)



#===========================================================================
    AGENTS: AUXILIARY
===========================================================================#

# Preferences
abstract type TipoPreferencias end
struct CRRA <: TipoPreferencias end
struct Preferencias{TP<:TipoPreferencias}
    β::Real             # Discount factor
    u::Function         # Utility function
    u′::Function        # Marginal utility
    inv_u′::Function    # Inverse marginal utility
    function Preferencias(::CRRA; β::Real, γ::Real)
        u = c::Real -> (γ ≈ 1.0) ? log(c) : c^(1.0-γ) / (1.0-γ)
        u′ = c::Real -> c^(-γ)
        inv_u′ = (u′) -> u′^(-1.0/γ)
        return new{CRRA}(β, u, u′, inv_u′)
    end
end

# State Indicex
abstract type AbstractStateIndices end
struct StateIndices <: AbstractStateIndices
    z::Vector{<:Int}
    a::Vector{<:Int}
end
Base.length(states::AbstractStateIndices) = length(states.a)
struct CombinedStateIndices <: AbstractStateIndices
    age::Vector{<:Int}
    z::Vector{<:Int}
    a::Vector{<:Int}
end

function combine(vec_states::Vector{<:AbstractStateIndices})
    ig = vcat([fill(i, length(states)) for (i, states) in enumerate(vec_states)]...)
    iz = vcat(getfield.(vec_states, :z)...)
    ia = vcat(getfield.(vec_states, :a)...)
    return CombinedStateIndices(ig, iz, ia)
end

# State variables
abstract type AbstractStateVariables end
Base.length(S::AbstractStateVariables) = length(S.a)

function assemble(Ss::Vector{<:Tsv}) where {Tsv <: AbstractStateVariables}
    function f_extract(S, fld)
        extracted = getfield(S, fld)
        return extracted isa AbstractVector ? extracted : fill(extracted, length(S))
    end
    return hcat([vcat(f_extract.(Ss, fld)...) for fld in fieldnames(Tsv)]...)
end
function combine(Ss::Vector{<:Tsv}) where {Tsv <: AbstractStateVariables}
    return Tsv(collect.(assemble(Ss) |> eachcol)...)
end

# Policy functions
mutable struct PolicyFunctions
    c::Vector{<:Real}
    a′::Vector{<:Real}
    # Constructor
    PolicyFunctions(c::Vector{<:Real}, a′::Vector{<:Real}) = new(c, a′)
    # Initialiser
    function PolicyFunctions(N::Int)
        c = Array{Float64}(undef, N)
        a′ = Array{Float64}(undef, N)
        return new(c, a′)
    end
end

function combine(Gs::Vector{<:PolicyFunctions})
    c = vcat(getfield.(Gs, :c)...)
    a′ = vcat(getfield.(Gs, :a′)...)
    return PolicyFunctions(c, a′)
end

# Life-cycle structure
function get_ages(; min_age::Int, max_age::Int, years_per_period::Int)
    return range(min_age, max_age, step=years_per_period)
end
get_life_cycle_parameters() = [:min_age, :max_age, :years_per_period]



#===========================================================================
    GROUPS OF AGENTS
===========================================================================#

abstract type AbstractHouseholds end
abstract type AgentGroup end
abstract type AbstractGenerationType end
struct StandardGen <: AbstractGenerationType end
struct Newby <: AbstractGenerationType end
struct Oldest <: AbstractGenerationType end
struct CombinedGen <: AbstractGenerationType end

struct Generation{Tg<:AbstractGenerationType} <: AgentGroup
    # Characteristics
    min_age::Int
    max_age::Int
    N::Int
    # Assets grid
    grid_a::AbstractGrid
    min_a′::Real
    # States and policy functions
    states::AbstractStateIndices
    S::AbstractStateVariables
    G::PolicyFunctions
    # Value function
    v::Vector{<:Real}
    # Other
    Q::SparseMatrixCSC
    distr::Vector{<:Real}
    euler_errors::Vector{<:Real}
end

# Initialiser
function Generation(
    type::AbstractGenerationType, min_age::Int, max_age::Int, grid_z, grid_a, min_a′::Real, ζ::Real
)
    # Matrix of state indices
    states = get_StateIndices(; N_z=size(grid_z), N_a=size(grid_a))
    # Number of agents in a generation
    N = size(states.a, 1)
    # State variables
    S = get_StateVariables(states, ζ, grid_z, grid_a)
    # Initialise policy functions
    G = PolicyFunctions(N)
    # Initialise value function
    vv = similar(S.z)
    # Initialise Q-transition matrix
    Q = spzeros(N, N)
    # Initialise distribution
    distr = fill(1/N, N)
    # Initialise euler errors
    euler_errs = similar(distr)
    # Return structure
    return Generation{typeof(type)}(min_age, max_age, N, grid_a, min_a′, states, S, G, vv, Q, distr, euler_errs)
end

function Generations(; ages::AbstractVector, tipo_a, grid_kwargs, grid_z, ζ_f::Function)::Vector{<:Generation}
    # Min and max ages
    min_ages = ages[1:(end-1)]
    max_ages = ages[2:end] .- 1
    avg_ages = (min_ages .+ max_ages) / 2
    # Number of generations
    N_g = length(min_ages)
    # Life-cycle productivity
    malla_ζ = [ζ_f(avg_ages[ig]) for ig in 1:N_g]
    malla_ζ .= malla_ζ / (sum(malla_ζ)/N_g)
        # Average productivity while working: normalisation to one
        # this works as population is equally distributed across generations (certain lifespan)
    # Age-dependent assets grids
    grids_a = CombinedGrid(tipo_a; ages=avg_ages, grid_kwargs...)
    min_a′ = grid_kwargs[:min]
    # Vector of generations
    gens = [Generation(StandardGen(), min_ages[ig], max_ages[ig], grid_z, grids_a[ig], min_a′, malla_ζ[ig]) for ig in 2:(N_g-1)]
    gens = vcat(Generation(Newby(), min_ages[1], max_ages[1], grid_z, grids_a[1], min_a′, malla_ζ[1]),
                gens,
                Generation(Oldest(), min_ages[end], max_ages[end], grid_z, grids_a[end], min_a′, malla_ζ[end]))
    return gens
end

# Methods
Base.size(g::Generation) = g.N
get_N_agents(gens::Vector{<:Generation}) = sum(assemble(gens, :N))
get_age_range(g::Generation) = string(g.min_age, "-", g.max_age)
get_preference_parameters() = [:tipo_pref, :β, :γ]
assemble(x, key::Symbol) = vcat(getproperty.(x, key)...)
assemble(x, key1::Symbol, key2::Symbol) = assemble(assemble(x, key1), key2)
assemble(gens::Vector{Generation}, f::Function, args...) = vcat([f(g, args...) for g in gens]...)
# Example: assemble(gens, :states, :z) will return a vector of all z states across generations.

# Combine generations
function combine(gens::Vector{<:Generation})
    # Assemble variables
    min_ages = assemble(gens, g -> fill(g.min_age, g.N))
    max_ages = assemble(gens, g -> fill(g.max_age, g.N))
    v = assemble(gens, :v)
    Q = sparse([1],[1],[NaN],1,1)
    distr = assemble(gens, :distr)
    euler_errors = assemble(gens, :euler_errors)
    # Other variables
    min_age = minimum(min_ages)
    max_age = maximum(max_ages)
    min_a′ = NaN  # This could be a vector of min_a for each agent, if needed
    # Structures
    grids_a = CombinedGrid([g.grid_a for g in gens])
    states = combine(assemble(gens, :states))
    S = combine(assemble(gens,:S))
    G = combine(assemble(gens,:G))
    # Create combined generation
    return Generation{CombinedGen}(min_age, max_age, length(S), grids_a, min_a′, states, S, G, v, Q, distr, euler_errors)
end
function combine(gens::Vector{<:Generation}, howmany::Int)
    howmany==1 && return gens  # don't do anything if howmany == 1
    return [combine(gens[i:(i+howmany-1)]) for i in 1:howmany:length(gens)]
end



#===========================================================================
    PRODUCERS
===========================================================================#

struct Firms
    α::Real             # Capital share
    δ::Real             # Depreciation rate
    F::Function         # Production function
    F′k::Function       # Marginal product of capital
    F′l::Function       # Marginal product of labor
    ratio_KL::Function  # Capital-labor ratio
    function Firms(; α::Real, δ::Real)
        F = (K, L)   ->         K^α     * L^(1-α)
        F′k = (K, L) ->    α  * K^(α-1) * L^(1-α)
        F′l = (K, L) -> (1-α) * K^α     * L^(-α)
        ratio_KL = (r) -> ( (r+δ)/α )^( 1.0/(α-1.0) )
        return new(α, δ, F, F′k, F′l, ratio_KL)
    end
end
get_firm_parameters() = [:α, :δ]



#===========================================================================
    GENERAL EQUILIBRIUM
===========================================================================#

mutable struct Prices
    r::Real
    w::Real
end



#===========================================================================
    ECONOMY
===========================================================================#

mutable struct TimeStructure
    years_per_period::Int
    years_cohort::Int
end

# Aggregates
mutable struct Aggregates
    A::Real     # aggregate savings
    A0::Real    # aggregate beginning-of-period assets
    K::Real     # aggregate capital
    L::Real     # aggregate labor
    Y::Real     # aggregate output
    C::Real     # aggregate consumption
end

struct Economía
    # Agents
    hh::AbstractHouseholds
    fm::Firms
    # Prices
    pr::Prices
    # Aggregates
    agg::Aggregates
    # Time structure
    time_str::TimeStructure
    # Basic initialiser
    function Economía(r_0::Real, hh::AbstractHouseholds, fm::Firms, years_per_period::Int)
        # Initialise prices
        pr = Prices(r_0, get_w(r_0, fm))
        # Initialise aggregates
        agg = get_aggregates(hh, fm, pr)
        # Create time structure
        time_str = TimeStructure(years_per_period, years_per_period)
        # Return the structure
        return new(hh, fm, pr, agg, time_str)
    end
end

function update_aggregates!(eco::Economía)::Nothing
    # Unpack
    @unpack hh, fm, pr = eco
    # Compute new aggregates
    agg = get_aggregates(hh, fm, pr)
    # Update values in eco structure
    eco.agg.A   = agg.A
    eco.agg.A0  = agg.A0
    eco.agg.K   = agg.K
    eco.agg.L   = agg.L
    eco.agg.Y   = agg.Y
    eco.agg.C   = agg.C
    return nothing
end



#===========================================================================
    ECONOMIC VARIABLES
===========================================================================#


abstract type EconomicVariable end
struct Savings <: EconomicVariable end
struct Assets <: EconomicVariable end
struct Consumption <: EconomicVariable end
struct Labour_Income <: EconomicVariable end
struct Interest_Rate <: EconomicVariable end
struct Capital <: EconomicVariable end
struct Labour <: EconomicVariable end

function get_economic_variable(key::Symbol)
    if key == :a′
        return Savings
    elseif key == :a
        return Assets
    elseif key == :c
        return Consumption
    elseif key == :incL
        return Labour_Income
    elseif key == :r
        return Interest_Rate
    elseif key == :K
        return Capital
    elseif key == :L
        return Labour
    else
        error("Unknown economic variable: $key")
    end
end
function get_var_string(key::Symbol)
    if key == :a′
        return "savings"
    elseif key == :a
        return "assets"
    elseif key == :c
        return "consumption"
    elseif key == :incL
        return "labour income"
    elseif key == :r
        return "interest rate"
    elseif key == :K
        return "capital"
    elseif key == :L
        return "labour"
    else
        error("Unknown economic variable: $key")
    end
end
function get_symbol(ev::EconomicVariable)
    if ev isa Savings
        return :a′
    elseif ev isa Assets
        return :a
    elseif ev isa Consumption
        return :c
    elseif ev isa Labour_Income
        return :incL
    elseif ev isa Interest_Rate
        return :r
    elseif ev isa Capital
        return :K
    elseif ev isa Labour
        return :L
    else
        error("Unknown economic variable: $ev")
    end
end



#===========================================================================
    STATISTICS
===========================================================================#

abstract type StatisticType end
struct Total <:StatisticType end
struct Share <:StatisticType end
struct Percentage <:StatisticType end
struct Mean <:StatisticType end
struct Probability <:StatisticType end

abstract type AbstractStatistic end
struct Stat{Ts<:StatisticType, Ev<:EconomicVariable} <: AbstractStatistic
    value::Real
    desc::String
    function Stat(::Ts, value::Real, keyvar::Symbol, desc::String) where {Ts<:StatisticType}
        new{Ts, get_economic_variable(keyvar)}(value, desc)
    end
end
struct StatDistr{Ts<:StatisticType, Ev<:EconomicVariable} <: AbstractStatistic
    values::Vector{<:Real}
    labels::Vector{<:String}
    desc::String
    function StatDistr(::Ts, values::Vector{<:Real}, labels::Vector{<:String}, keyvar::Symbol, desc::String) where {Ts<:StatisticType}
        new{Ts, get_economic_variable(keyvar)}(values, labels, desc)
    end
end
struct StatFutureDistr{Ts<:StatisticType, Ev<:EconomicVariable} <: AbstractStatistic
    values::Vector{<:Real}
    labels::Vector{<:String}
    desc::String
    periods::Int
    initial_group::String
    function StatFutureDistr(::Ts, values::Vector{<:Real}, labels::Vector{<:String}, keyvar::Symbol, desc::String, periods::Int, initial_group::String) where {Ts<:StatisticType}
        new{Ts, get_economic_variable(keyvar)}(values, labels, desc, periods, initial_group)
    end
end

# Methods
Base.zip(sd::StatDistr{<:StatisticType}) = zip(sd.values, sd.labels)
Base.zip(sd::StatFutureDistr{<:StatisticType}) = zip(sd.values, sd.labels)
Base.length(sd::StatDistr{<:StatisticType}) = length(sd.values)
Base.size(sd::StatDistr{<:StatisticType}) = length(sd.values)
Base.size(sd::StatFutureDistr{<:StatisticType}) = length(sd.values)
Base.sum(sd::StatDistr{<:Percentage}) = sum(sd.values)
Base.sum(sd::StatDistr{<:Probability}) = sum(sd.values)
get_economic_variable(::Stat{<:StatisticType, Ev}) where {Ev<:EconomicVariable} = Ev
get_economic_variable(::StatDistr{<:StatisticType, Ev}) where {Ev<:EconomicVariable} = Ev
get_symbol(::Stat{<:StatisticType, Ev}) where {Ev<:EconomicVariable} = get_symbol(Ev())
get_symbol(::StatDistr{<:StatisticType, Ev}) where {Ev<:EconomicVariable} = get_symbol(Ev())
function describe(ev::EconomicVariable)
    desc = ev |> nameof |> string 
    replace!("_" => " ", desc)
    return lowercase(desc)
end