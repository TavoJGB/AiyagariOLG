using Aiyagari
import Aiyagari: err_euler, err_budget, get_borrowing_constrained, get_r
using Aiyagari.LinearAlgebra
using Aiyagari.Parameters

using Plots
using Test

# Settings
tol = 1e-6

# Run benchmark economy
    # Settings
    hlds, prod, her, cfg = build_model();
    # Steady-state computations
    eco = steady(hlds, prod, her, cfg; r_0=0.04);
    agg = Aggregates(eco)                                       # aggregate variables

@testset "Steady State" begin
    @unpack hh, fm, pr, distr, Q = eco
    @unpack states, ind, process_z = her
    @unpack S, G = hh
    @unpack A, A0, K, L, C, Y = agg
    
    # Q-MATRIX
    # Total transition probabilities for each combination of states (each column) is 1
    @test maximum(abs.(sum(Q, dims=1) .- 1)) < tol

    # STATIONARY DISTRIBUTION
    # Total population is normalised to 1
    @test abs(sum(eco.distr) - 1) < tol
    # SS distribution by productivity level
    ss_z_dist = vcat([sum(distr[indJ]) for indJ in eachcol(states[:,ind.z].==(1:size(process_z))')]...)
    @test maximum(abs.(ss_z_dist - process_z.ss_dist)) < tol

    # HOUSEHOLD'S PROBLEM
    # Euler equation (only applies to unconstrained households)
    if false
        errs_eu = err_euler(eco)
        constrained = get_borrowing_constrained(eco, her; tol=0)
        @test sum(distr .* (abs.(errs_eu).>tol) .* (.!constrained)) < tol
        plot(hh.G.a′[.!constrained], errs_eu[.!constrained], seriestype=:scatter, title="Euler errors", xlabel="a′", ylabel="error", label="", legend=false)
        plot(hh.S.a[.!constrained], errs_eu[.!constrained], seriestype=:scatter, title="Euler errors", xlabel="a", ylabel="error", label="", legend=false)
    end
    # Budget constraint
    @test maximum(abs.(err_budget(eco))) < tol

    # MARKETS
    # Capital market: households' side
    @test abs(dot(G.a′, eco.distr) - A) < tol
    # Capital market: firms' side
    @test abs(fm.ratio_KL(pr.r)*L - K) < tol
    # Capital market: clearing
    @test abs(A - K) < tol
    # Capital market: stationarity
    @test abs(A - A0) < tol
    # Goods market: clearing
    @test abs(A + C - (1-fm.δ)A0 - Y) < tol
end