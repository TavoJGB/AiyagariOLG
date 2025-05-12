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
    TOOLS
===========================================================================#

# Matrix of state variables
function state_matrix(N_z::Ti, N_a::Ti, ind::NamedTuple) where {Ti<:Integer}
    # Initialise matrix
    states = Matrix{Ti}(undef, N_a*N_z, 2)
    # Fill it
    states[:,ind.z] = kron(1:N_z, ones(N_a))
    states[:,ind.a] = repeat(1:N_a, N_z)
    # Return it
    return states
end

# Main structure
struct Herramientas
    # State variables
    process_z::MarkovProcess
    grid_a::AbstractGrid
    states::Matrix{<:Real}
    ind::NamedTuple
    function Herramientas(; process_z::MarkovProcess, grid_a::AbstractGrid)
        ind = (z=1, a=2)
        states = state_matrix(size(process_z), size(grid_a), ind)
        return new(process_z, grid_a, states, ind)
    end
end

# Methods
grids(her::Herramientas) = her.process_z.grid, her.grid_a



#===========================================================================
    HOUSEHOLDS
===========================================================================#

# Preferences
abstract type TipoPreferencias end
struct CRRA <: TipoPreferencias end
struct Preferencias{TP<:TipoPreferencias}
    β::Real             # Discount factor
    u::Function         # Utility function
    u′::Function        # Marginal utility
    inv_u′::Function    # Inverse marginal utility
    function Preferencias(::CRRA, β::Real, γ::Real)
        u = c::Real -> (γ ≈ 1.0) ? log(c) : c^(1.0-γ) / (1.0-γ)
        u′ = c::Real -> c^(-γ)
        inv_u′ = (u′) -> u′^(-1.0/γ)
        return new{CRRA}(β, u, u′, inv_u′)
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

# Main structure
struct Households
    N::Int
    pref::Preferencias
    S::StateVariables
    G::PolicyFunctions
    function Households(her::Herramientas; tipo_pref, β::Real, γ::Real)
        # Unpack
        @unpack states, ind = her
        grid_z, grid_a = grids(her)
        # Number of agents
        N = size(states, 1)
        # Preferences
        pref = Preferencias(tipo_pref, β, γ)
        # State variables
        zz = get_node.(Ref(grid_z), states[:, ind.z])
        aa = get_node.(Ref(grid_a), states[:, ind.a])
        S = StateVariables(zz, aa)
        # Policy functions
        G = PolicyFunctions(N)
        return new(N, pref, S, G)
    end
end
get_household_parameters() = [:tipo_pref, :β, :γ]



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
    STATISTICS
===========================================================================#

abstract type StatisticType end
struct Share <:StatisticType end
struct Percentage <:StatisticType end
struct Mean <:StatisticType end

abstract type AbstractStatistic end
struct Stat{Ts<:StatisticType} <: AbstractStatistic
    value::Float64
    desc::String
    function Stat(::Ts, value::Float64, desc::String) where {Ts<:StatisticType}
        new{Ts}(value, desc)
    end
end
struct StatDistr{Ts} <: AbstractStatistic where {Ts<:StatisticType}
    values::Vector{<:Float64}
    labels::Vector{<:String}
    desc::String
    function StatDistr(::Ts, values::Vector{<:Float64}, labels::Vector{<:String}, desc::String) where {Ts<:StatisticType}
        new{Ts}(values, labels, desc)
    end
end
Base.zip(sd::StatDistr{<:StatisticType}) = zip(sd.values, sd.labels)
Base.size(sd::StatDistr{<:StatisticType}) = length(sd.values)
Base.sum(sd::StatDistr{<:Percentage}) = sum(sd.values)