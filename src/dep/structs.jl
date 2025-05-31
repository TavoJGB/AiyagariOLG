#===========================================================================
    SOLVERS
===========================================================================#

# Solvers
abstract type SolverType end
struct LinearJumps <: SolverType end
struct EGM <: SolverType end
struct POWM <: SolverType end

# Parameters of the solvers
abstract type SolverParameters end
@with_kw struct BasicSolverParameters <: SolverParameters
    objective::String
    tol::Real
    maxit::Int
end
SolverParameters(::SolverType; kwargs...) = BasicSolverParameters(; kwargs...)
get_solver_parameters(::SolverType) = [:objective, :tol, :maxit]
@with_kw struct LinearJumpsParameters <: SolverParameters
    objective::String
    tol::Real
    wgt::Real
    maxit::Int
end
SolverParameters(::LinearJumps; kwargs...) = LinearJumpsParameters(; kwargs...)
get_solver_parameters(::LinearJumps) = [:objective, :tol, :wgt, :maxit]

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
    cfg_hh::Solver
    cfg_graph::GraphConfig
    # Years per model period
    years_per_period::Int
end



#===========================================================================
    GRIDS
===========================================================================#

abstract type AbstractGrid end

struct BasicGrid <: AbstractGrid
    N::Int
    nodes::Vector{<:Real}
    BasicGrid(nodes) = new(length(nodes), nodes) 
end
get_BasicGrid_parameters() = [:nodes]
struct LinearGrid <: AbstractGrid
    N::Int
    min::Real
    max::Real
    nodes::Vector{<:Real}
end
function _LinearGrid(N::Int, min::Real, max::Real)
    nodes = range(min, max, length=N) |> collect
    return LinearGrid(N, min, max, nodes)
end
get_LinearGrid_parameters() = [:N, :min, :max]
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
get_CurvedGrid_parameters() = [:N, :curv, :min, :max]

# Grid methods
Base.size(grid::AbstractGrid) = grid.N
Base.length(grid::AbstractGrid) = grid.N
function get_node(grid::AbstractGrid, i::Int)
    return grid.nodes[i]
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
    ss_dist::Vector{<:Real}     # Steady state distribution
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
    return DiscreteAR1(ρ, σ, BasicGrid(nodes), Π, ss_dist)
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
    function StateIndices(; N_z::Ti, N_a::Ti) where {Ti<:Integer}
        return new(kron(1:N_z, ones(Ti, N_a)), repeat(1:N_a, N_z))
    end
end
struct CombinedStateIndices <: AbstractStateIndices
    age::Vector{<:Int}
    z::Vector{<:Int}
    a::Vector{<:Int}
end

# State variables
struct StateVariables
    z::Vector{<:Real}
    a::Vector{<:Real}
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

# Life-cycle structure
function get_ages(; min_age::Int, max_age::Int, years_per_period::Int)
    return range(min_age, max_age, step=years_per_period)
end
get_life_cycle_parameters() = [:min_age, :max_age, :years_per_period]



#===========================================================================
    GROUPS OF AGENTS
===========================================================================#

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
    # States and policy functions
    states::AbstractStateIndices
    S::StateVariables
    G::PolicyFunctions
    # Value function
    v::Vector{<:Real}
    # Other
    Q::SparseMatrixCSC
    distr::Vector{<:Real}
    euler_errors::Vector{<:Real}
end
# Initialiser
function Generation(type::AbstractGenerationType, min_age::Int, max_age::Int, grid_z::AbstractGrid, grid_a::AbstractGrid)
    # Matrix of state indices
    states = StateIndices(; N_z=size(grid_z), N_a=size(grid_a))
    # Number of agents in a generation
    N = size(states.a, 1)
    # State variables
    zz = get_node.(Ref(grid_z), states.z)
    aa = get_node.(Ref(grid_a), states.a)
    S = StateVariables(zz, aa)
    # Initialise policy functions
    G = PolicyFunctions(N)
    # Initialise value function
    vv = similar(zz)
    # Initialise Q-transition matrix
    Q = spzeros(N, N)
    # Initialise distribution
    distr = fill(1/N, N)
    # Initialise euler errors
    euler_errs = similar(distr)
    # Return structure
    return Generation{typeof(type)}(min_age, max_age, N, states, S, G, vv, Q, distr, euler_errs)
end
# Methods
Base.size(g::Generation) = g.N
get_N_agents(gens::Vector{Generation}) = sum(assemble(gens, :N))
get_age_range(g::Generation) = string(g.min_age, "-", g.max_age)

# Combine generations
function combine(gens::Vector{Generation})
    # Assemble variables
    min_ages = assemble(gens, g -> fill(g.min_age, g.N))
    max_ages = assemble(gens, g -> fill(g.max_age, g.N))
    iz = assemble(gens, :states, :z)
    ia = assemble(gens, :states, :a)
    z = assemble(gens, :S, :z)
    a = assemble(gens, :S, :a)
    c = assemble(gens, :G, :c)
    a′ = assemble(gens, :G, :a′)
    v = assemble(gens, :v)
    Q = sparse([1],[1],[NaN],1,1)
    distr = assemble(gens, :distr)
    euler_errors = assemble(gens, :euler_errors)
    # Other variables
    min_age = minimum(min_ages)
    max_age = maximum(max_ages)
    N = length(z)
    # Structures
    states = CombinedStateIndices(min_ages, iz, ia)
    S = StateVariables(z, a)
    G = PolicyFunctions(c, a′)
    # Create combined generation
    return Generation{CombinedGen}(min_age, max_age, N, states, S, G, v, Q, distr, euler_errors)
end
function combine(gens::Vector{Generation}, howmany::Int)
    howmany==1 && return gens  # don't do anything if howmany == 1
    return [combine(gens[i:(i+howmany-1)]) for i in 1:howmany:length(gens)]
end



#===========================================================================
    HOUSEHOLDS
===========================================================================#

struct Households
    N::Int
    gens::Vector{Generation}
    pref::Preferencias
    process_z::MarkovProcess
    grid_a::AbstractGrid
    # Constructors
    function Households(;
        ages::AbstractVector, tipo_pref, process_z::MarkovProcess, grid_a::AbstractGrid, kwargs...
    )
        # Unpack
        grid_z = process_z.grid
        # Preferences
        pref = Preferencias(tipo_pref; kwargs...)
        # Vector of generations
        gens = [Generation(StandardGen(), min_age, max_age-1, grid_z, grid_a) for (max_age, min_age) in zip_forward(ages[2:(end-1)])]
        gens = vcat(Generation(Newby(), ages[1], ages[2]-1, grid_z, grid_a),
                    gens,
                    Generation(Oldest(), ages[end-1], ages[end]-1, grid_z, grid_a))
        # Total number of agents
        N = get_N_agents(gens)
        # Return structure
        return new(N, gens, pref, process_z, grid_a)
    end
end

# Methods
get_preference_parameters() = [:tipo_pref, :β, :γ]
grids(hh::Households) = hh.process_z.grid, hh.grid_a
assemble(x, key::Symbol) = vcat(getproperty.(x, key)...)
assemble(x, key1::Symbol, key2::Symbol) = assemble(assemble(x, key1), key2)
assemble(gens::Vector{Generation}, f::Function, args...) = vcat([f(g, args...) for g in gens]...)
# Example: assemble(gens, :states, :z) will return a vector of all z states across generations.



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

mutable struct Aggregates
    A::Real     # aggregate savings
    A0::Real    # aggregate beginning-of-period assets
    K::Real     # aggregate capital
    L::Real     # aggregate labor
    Y::Real     # aggregate output
    C::Real     # aggregate consumption
    function Aggregates(hh::Households, fm::Firms, pr::Prices)
        # Unpack
        @unpack gens = hh
        @unpack ratio_KL, F = fm
        # Assemble states and policy functions
        a′ = assemble(gens, :G, :a′)
        c = assemble(gens, :G, :c)
        a = assemble(gens, :S, :a)
        z = assemble(gens, :S, :z)
        # Distribution
        distr = assemble(gens, :distr)
        # Households
        A = dot(distr, a′)
        A0 = dot(distr, a)
        C = dot(distr, c)
        L = dot(distr, z)
        # Firms
        K = ratio_KL(pr.r) * L
        Y = F(K, L)
        return new(A, A0, K, L, Y, C)
    end
end

struct Economía
    # Agents
    hh::Households
    fm::Firms
    # Prices
    pr::Prices
    # Aggregates
    agg::Aggregates
    # Time structure
    time_str::TimeStructure
    # Basic initialiser
    function Economía(r_0::Real, hh::Households, fm::Firms, years_per_period::Int)
        # Initialise prices
        pr = Prices(r_0, get_w(r_0, fm))
        # Initialise aggregates
        agg = Aggregates(hh, fm, pr)
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
    agg = Aggregates(hh, fm, pr)
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
    else
        error("Unknown economic variable: $ev")
    end
end



#===========================================================================
    STATISTICS
===========================================================================#

abstract type StatisticType end
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