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
end
function _GraphConfig(; plotsiz::String, kwargs...)
    return GraphConfig(; plotsiz=parse.(Int, split(plotsiz,"x")), kwargs...)
end

struct Configuration
    cfg_r::Solver
    cfg_hh::Solver
    cfg_distr::Solver
    cfg_graph::GraphConfig
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
struct StateIndices
    z::Vector{<:Int}
    a::Vector{<:Int}
    function StateIndices(; N_z::Ti, N_a::Ti) where {Ti<:Integer}
        return new(kron(1:N_z, ones(Ti, N_a)), repeat(1:N_a, N_z))
    end
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
    function PolicyFunctions(N::Int)
        c = Array{Float64}(undef, N)
        a′ = Array{Float64}(undef, N)
        return new(c, a′)
    end
end



#===========================================================================
    HOUSEHOLDS
===========================================================================#

struct Households
    N::Int
    pref::Preferencias
    states::StateIndices
    S::StateVariables
    G::PolicyFunctions
    process_z::MarkovProcess
    grid_a::AbstractGrid
    function Households(;
        tipo_pref, process_z::MarkovProcess, grid_a::AbstractGrid, kwargs...
    )
        # Unpack
        grid_z = process_z.grid
        # Matrix of state indices
        states = StateIndices(; N_z=size(process_z), N_a=size(grid_a))
        # Number of agents
        N = size(states.a, 1)
        # Preferences
        pref = Preferencias(tipo_pref; kwargs...)
        # State variables
        zz = get_node.(Ref(grid_z), states.z)
        aa = get_node.(Ref(grid_a), states.a)
        S = StateVariables(zz, aa)
        # Policy functions
        G = PolicyFunctions(N)
        return new(N, pref, states, S, G, process_z, grid_a)
    end
end
get_preference_parameters() = [:tipo_pref, :β, :γ]
grids(hh::Households) = hh.process_z.grid, hh.grid_a



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

struct Economía
    # Agents
    hh::Households
    fm::Firms
    # Prices
    pr::Prices
    # Other
    Q::SparseMatrixCSC
    distr::Vector{<:Real}
    # Basic initialiser
    function Economía(r_0::Tr, hh::Households, fm::Firms) where {Tr<:Real}
        @unpack N = hh
        # Initialise prices and policy functions
        pr = Prices(r_0, get_w(r_0, fm))
        guess_G!(hh, pr)
        # Initialise Q-transition matrix
        Q = spzeros(N, N)
        # Initialise distribution
        distr = ones(Tr, N) / N
        # Return the structure
        return new(hh, fm, pr, Q, distr)
    end
end

struct Aggregates
    A::Real     # aggregate savings
    A0::Real    # aggregate beginning-of-period assets
    K::Real     # aggregate capital
    L::Real     # aggregate labor
    Y::Real     # aggregate output
    C::Real     # aggregate consumption
    function Aggregates(eco::Economía)
        # Unpack
        @unpack pr, hh, fm, distr = eco
        @unpack c, a′ = hh.G
        @unpack z, a = hh.S
        @unpack ratio_KL, F = fm
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