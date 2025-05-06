using Aiyagari

# Settings
model = build_model(; N_z=8, N_a=100);

# Compute steady state
eco = steady(model...; r_0=0.04);