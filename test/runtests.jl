# Additional packages
using Plots
using Test

# Loading main module and dependencies
using Aiyagari
import Aiyagari: err_euler, err_budget, get_borrowing_constrained, get_r
using Aiyagari.LinearAlgebra
using Aiyagari.Parameters
include("./dep/ss_test.jl")

# Settings
tol = 1e-6


# DEFAULT ECONOMY
model_dft = build_model(; save_pars=false);
# Steady-state computations
eco = steady(model_dft...; r_0=0.04);
# Testing
@testset "BENCHMARK SIMULATION: Steady State" begin
    ss_test(eco, model_dft.her; tol)
end


# MOST RECENT SIMULATION
model_last = build_model("../parameters/latest_simulation.csv"; save_pars=false);
# Steady-state computations
eco = steady(model_last...; r_0=0.04);
# Testing
@testset "LATEST SIMULATION: Steady State" begin
    ss_test(eco, model_last.her; tol)
end