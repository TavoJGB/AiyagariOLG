# Additional packages
using Plots
using Test

# Loading main module and dependencies
using AiyagariOLG
import AiyagariOLG: err_euler, err_budget, get_borrowing_constrained, get_r, zip_backward, zip_forward
using AiyagariOLG.LinearAlgebra
using AiyagariOLG.Parameters
include("./dep/utils_test.jl")
include("./dep/ss_test.jl")

# Settings
tol = 1e-6


# TEST ECONOMY
# Test that we keep getting the same results for a given set of parameters
model_tst = build_model("../Simulations/parameters/test_parameters.csv"; save_pars=false);
# Steady-state computations
eco_tst = steady(model_tst...; r_0=0.04);
# Testing
@testset "BENCHMARK SIMULATION: Steady State" begin
    ss_test(eco_tst; tol)
    quantiles_test(eco_tst; tol)
    annualise!(eco_tst)
    @test compare_results(eco_tst; nq=5, top=0.1) < tol
end


# MOST RECENT SIMULATION
# Test that all equilibrium conditions are satisfied for the most recent simulation
model_last = build_model("../Simulations/parameters/latest_simulation.csv"; save_pars=false);
# Steady-state computations
eco_last = steady(model_last...; r_0=0.04);
# Testing
@testset "LATEST SIMULATION: Steady State" begin
    ss_test(eco_last; tol)
    quantiles_test(eco_last; tol)
end