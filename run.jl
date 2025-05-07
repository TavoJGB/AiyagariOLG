using Aiyagari

# Settings
model = build_model(; N_z=5, N_a=500, max_a=200);

# Compute steady state
eco = steady(model...; r_0=0.04);

# Display steady state
ss_summary(eco, model.her)