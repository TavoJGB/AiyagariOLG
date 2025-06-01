#===========================================================================
    SOLVERS
===========================================================================#

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
            println("#$it \t Error: $err \t Guess: $(round(100*guess,digits=2)) % vs $(round(100*newguess,digits=2)) %")
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
    ZIP FORWARD AND BACKWARDS
    Methods to get pairs of subsequent elements in a vector
    (f.e. in a vector of generations)
===========================================================================#

# Methods to get pairs of subsequent elements in a vector
# (f.e. in a vector of generations)
zip_backward(x::AbstractVector) = zip(x[(end-1):-1:1], x[end:-1:2])
zip_forward(x::AbstractVector) = zip(x[2:end], x[1:(end-1)])
# Test:
# for (g, g′) in zip_backward(gens)
#     println("Age g: $(get_age_range(g)), Age g′: $(get_age_range(g′))")
# end
# for (g, g_prev) in zip_forward(gens)
#     println("Age g: $(get_age_range(g)), Age g_prev: $(get_age_range(g_prev))")
# end