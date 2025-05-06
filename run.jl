using Aiyagari

# Settings
model = build_model(; proc_z_N=5, grid_a_N=100);

# Compute steady state
eco = steady(model...; r_0=0.04);