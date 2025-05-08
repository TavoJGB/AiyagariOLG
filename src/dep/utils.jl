#===========================================================================
    SOLVERS
===========================================================================#

# EGM
function solve!(solver::Solver{<:EGM}, args...)
    EGM!(solver.p, args...)
end
function EGM!(pars::SolverParameters, get_guess::Function, main, iterator!::Function, args...)
    @unpack tol, maxit = pars
    for it in 1:maxit
        # Initial guess
        guess = get_guess(main)
        # Iteration
        iterator!(main, args...)
        # Implied value
        implied = get_guess(main)
        # Check convergence
        if maximum(abs.(implied .- guess)) < tol
            # println(pars.objective, " converged in $it iterations")
            return nothing
        end
    end
    error("EGM did not converge")
end

# POWM
function solve(solver::Solver{<:POWM}, args...)
    return POWM(solver.p, args...)
end
function POWM(pars::SolverParameters, Q::AbstractMatrix)
    N = size(Q)[1]
    @unpack maxit, tol = pars
    _, x = powm!(Q, ones(N), maxiter=maxit, tol=tol)
    # returns the approximate largest eigenvalue λ of M and one of its eigenvector
    return x / sum(x)
end

# Linear Jumps: update to the linear combination of guess and implied values
function solve!(solver::Solver{<:LinearJumps}, args...)
    LinearJumps!(solver.p, args...)
end
function LinearJumps!(pars::SolverParameters, guess::Real, iterator!::Function, args...)
    @unpack tol, maxit, wgt = pars
    for it in 1:maxit
        # Iteration
        err, newguess = iterator!(guess, args...)
        # Check convergence
        if err < tol
            println(pars.objective, " converged in $it iterations")
            return nothing
        else
            # Update guess
            println("#$it: error = $err. Guess: $guess vs $newguess")
            guess = wgt*guess + (1-wgt)*newguess
        end
    end
    error("LinearJumps did not converge")
end



#===========================================================================
    INTERPOLATION: getting weights
===========================================================================#

abstract type InterpolationMethod end
struct Cap <: InterpolationMethod end
struct Extrapolate <: InterpolationMethod end

# Main method
function get_weights(::Extrapolate, x::Real, y::Vector{<:Real})

    # The function returns three vectors:
    # - lower: position in y of the nearest element below x.
    # - upper: position in y of the nearest element above x.
    # - weight: weight of lower in the linear combination that gives x as a function of y[lower] and y[upper].

    # Finding elements in y immediately above and below x
        # Number of elements in y:
            sizY = size(y,1)
        # Find lower neighbour in y
            lower = searchsortedlast(y, x)
            #lower is the largest index such that y[lower]≤x (and hence y[lower+1]>x). Returns 0 if x≤y[1]. y sorted.
        # Elements beyond the boundaries of y
            lower = clamp(lower, 1, sizY-1)
        # Corresponding upper neighbour
            upper = lower+1
    # Computing the weight of the element below
        weight = (y[upper] - x) / (y[upper] - y[lower])
        # the weight for the upper element is (1 - weight)
    # returns interpolated value and corresponding index
    return lower, upper, weight
end
function get_weights(::Cap, x::Real, y::Vector{<:Real})
    lower, upper, weight = get_weights(Extrapolate(), x, y)
    weight = clamp(weight, 0, 1)
    return lower, upper, weight
end

# Alternative method where I provide the position of the lower element,
# and the function returns the position of the upper element and the weight
function get_weights(::Extrapolate, x::Real, y::Vector{<:Real}, lower::Int)
    upper = lower+1
    weight = (y[upper] - x) / (y[upper] - y[lower])
    return upper, weight
end
function get_weights(::Cap, x::Real, y::Vector{<:Real}, lower::Int)
    upper, weight = get_weights(Extrapolate(), x, y, lower)
    weight = clamp(weight, 0, 1)
    return upper, weight
end

# Particular case: grid specification with curvature parameter curv
function get_weights(metodo::InterpolationMethod, x::Real, g::CurvedGrid)
    # Preliminaries
    @unpack N, curv, min, max, nodes = g
    Dx  = 1.0/(N-1)        # step in the grid
    Ti = typeof(N)
    # Intermediate transformation
    x_hat = ((x - min) / (max - min)) ^ curv
    n_hat = 1.0 + (x_hat / Dx)
    # Index of lower neighbour in grid for each a
    lower = floor(Ti, n_hat)
    lower = clamp(lower, 1, N-1)    
    # Upper neighbour and weight of lower neighbour
    upper, weight = get_weights(metodo, x, g.nodes, lower)
    return lower, upper, weight
end

# VECTORISED methods
function get_weights(::Extrapolate, x::Vector{<:Real}, y::Vector{<:Real}, lower::Vector{<:Int})
    upper = lower.+1
    weight = (y[upper] - x) ./ (y[upper] - y[lower])
    return upper, weight
end
function get_weights(::Cap, x::Vector{<:Real}, y::Vector{<:Real}, lower::Vector{<:Int})
    upper, weight = get_weights(Extrapolate(), x, y, lower)
    weight = clamp.(weight, 0, 1)
    return upper, weight
end
function get_weights(metodo, x::Vector{<:Real}, y::Vector{<:Real})
    # Number of elements in each vector:
    sizX = size(x,1)
    sizY = size(y,1)
    # Initialise vector
    lower = Array{Int64}(undef, sizX)
    # Find lower elements for each of them
    lower = clamp.(searchsortedlast.(Ref(y), x), 1, sizY-1)
    upper, weight = get_weights(metodo, x, y, lower)
    return lower, upper, weight
end
function get_weights(metodo, x::Vector{<:Real}, g::CurvedGrid)
    # Preliminaries
    @unpack N, curv, min, max, nodes = g
    Dx  = 1.0/(N-1)        # step in the grid
    Ti = typeof(N)
    # Intermediate transformation
    x_hat = ((x .- min) / (max - min)) .^ curv
    n_hat = 1.0 .+ (x_hat / Dx)
    # Index of lower neighbour in grid for each a
    lower = floor.(Ti, n_hat)
    lower = clamp.(lower, 1, N-1)    
    # Upper neighbour and weight of lower neighbour
    upper, weight = get_weights(metodo, x, g.nodes, lower)
    return lower, upper, weight
end



#===========================================================================
    INTERPOLATION: obtaining interpolated value
===========================================================================#

function interpLinear(
    x::Real, y::Vector{<:Real}, z::Vector{<:Real};
    metodo=Extrapolate()
)
    lower, upper, weight = get_weights(metodo, x, y)
    # returns interpolated value
    return weight*z[lower] + (1 - weight)*z[upper]
end
function interpLinear(
    x::Real, g::CurvedGrid, z::Vector{<:Real};
    metodo=Extrapolate()
)
    lower, upper, weight = get_weights(metodo, x, g)
    # returns interpolated value
    return weight*z[lower] + (1 - weight)*z[upper]
end

# VECTORISED methods
function interpLinear(
    x::Vector{<:Real}, y, z::Vector{<:Real};
    metodo=Extrapolate()
)
    lower, upper, weight = get_weights(metodo,x,y)
    return weight.*z[lower] .+ (1 .- weight).*z[upper]
end



#===========================================================================
    MARKOV PROCESSES
===========================================================================#

# ROUWENHORST
# Source: https://github.com/pereiragc/rouwenhorst/blob/master/rouwenhorst.jl

# # Usage example:
# npts = 10
# rho = 0.9
# sig = 0.01
# gridpts, transition = rouwenhorst(npts, rho, sig)


function RouwenNormalize!(A)
    A[2:end-1, :] ./= 2
    nothing
end
function rouwenupdate!(Πold, Πnew, p, q)
    Nold = size(Πold, 1)
    Πnew[1:Nold, 1:Nold] = p*Πold
    Πnew[1:Nold, 2:(Nold+1)] .+= (1-p)*Πold
    Πnew[2:(Nold+1), 1:Nold] .+= (1-q)*Πold
    Πnew[2:(Nold+1), 2:(Nold+1)] .+= q*Πold
    nothing
end
function rouwenmat(N, p, q)
    initmat = [p 1-p;1-q q]
    if N==2
        ret = initmat
    else
        currmat = initmat
        for n = 3:N
            nextmat = fill(0., (n, n))
            rouwenupdate!(currmat, nextmat, p, q)
            currmat = nextmat
            RouwenNormalize!(currmat)
        end
        ret = currmat
    end
    return(ret)
end
function rouwenhorst(npts, ρ, σ)
    # Discretizes
    # y′ = ρ y +  σ ϵ

    ω = σ/sqrt(1-ρ^2) # Long run std dev
    # println("Long run variance: $ω")
    q = (1+ρ)/2

    points = range(-ω*sqrt(npts - 1),
                   ω*sqrt(npts-1),
                   length=npts)

    # points = GenGrid(uniform(npts, bds))
    Π = rouwenmat(npts, q, q)
    return points, Π
end



#===========================================================================
    NAMED TUPLES
===========================================================================#

function subset_namedtuple(nt::NamedTuple, substr::String; typesubstr::String="Prefix")
    if typesubstr == "Prefix"
        subset = nt[filter(key -> startswith(String(key), substr), keys(nt))]
    elseif typesubstr == "Suffix"
        subset = nt[filter(key -> endswith(String(key), substr), keys(nt))]
    else
        error("typesubstr must be either 'Prefix' or 'Suffix'")
    end
    newkeys = Symbol.(replace.(String.(keys(subset)), substr => ""))
    newvals = values(subset)
    return NamedTuple(newkeys .=> newvals)
end



#===========================================================================
    IDENTIFICATION OF AGENTS
===========================================================================#

function identify_group(var::Vector{<:Real}, crit::Function)
    return crit.(var)
end
function identify_group(var::Vector{<:Real}, crit::Int)
    return identify_group(var, x -> x.==crit)
end
function identify_group(her::Herramientas, keyvar::Symbol, crit::Function)
    @unpack states, ind = her
    return crit.(states[:, ind[keyvar]])
end
function identify_group(her::Herramientas, keyvar::Symbol, crit::Int)
    return identify_group(her, keyvar, x -> x.==crit)
end
function identify_group(G::PolicyFunctions, keyvar::Symbol, crit::Function)
    return crit.(getproperty(G, keyvar))
end

# Borrowing contrained agents (both beggining and end of period)
function get_borrowing_constrained(a′, min_a, tol::Real)
    return a′ .<= min_a + tol
end
function get_borrowing_constrained(
    eco::Economía, her::Herramientas;
    tol=(her.grid_a.nodes[2]-her.grid_a.nodes[1])/2
)
    return get_borrowing_constrained(eco.hh.G.a′, her.grid_a.min, tol)
end
function get_borrowing_constrained(her::Herramientas)
    return identify_group(her, :a, 1)
end