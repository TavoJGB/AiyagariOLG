module ExtSolverEGM

#===========================================================================
    HOUSEHOLD PROBLEM
===========================================================================#

function EGM_savings!(gg::Generation{<:Oldest}, args...)::Nothing
    gg.G.a′ .= 0
    return nothing
end
function EGM_savings!(
    gg::Generation, pr::Prices, c′::Vector{<:Real}, grid_a′::AbstractGrid,
    pref::Preferencias, process_z::MarkovProcess
)::Nothing
    # Unpack
    @unpack N, S, states = gg
    malla_a = gg.grid_a.nodes
    malla_a′ = grid_a′.nodes
    # Initialise policy function for savings
    a_EGM = similar(c′)
    # Implied consumption and assets
    c_imp = c_euler(pref, c′, process_z.Π, pr.r, N, size(grid_a′))
    a_imp = a_budget(pr, c_imp, S)
    # Invert to get policy function for savings
    for zz=1:size(process_z)
        ind_z = (states.z .== zz)
        a_EGM[ind_z] = interpLinear(malla_a, a_imp[ind_z], malla_a′)
    end
    # Policy function bounds
    @. a_EGM = clamp(a_EGM, grid_a′.min, grid_a′.max)
    # Update policy function
    gg.G.a′ = a_EGM
    return nothing
end
function EGM_consumption!(gg::Generation, pr::Prices)::Nothing
    gg.G.c = budget_constraint(gg.G.a′, pr, gg.S)
    return nothing
end



#===========================================================================
    ITERATION
===========================================================================#

function EGM_iter!(gg::Generation, pr::Prices, args...)::Nothing
    EGM_savings!(gg, pr, args...)
    EGM_consumption!(gg, pr)
    return nothing
end



#===========================================================================
    MAIN FUNCTION
===========================================================================#

function EGM!(pars::SolverParameters, iterator!::Function, get_guess::Function, main, args...)
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
function AiyagariOLG.solve!(solver::Solver{<:LinearJumps}, args...)
    EGM!(solver.p, EGM_iter!, args...)
end



#===========================================================================
    END OF MODULE
===========================================================================#

end