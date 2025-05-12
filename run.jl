using Aiyagari

# Settings
model = build_model(; N_z=5, N_a=500);

# Compute steady state
eco = steady(model...; r_0=0.04);

# Display steady state
ss_analysis(eco, model.her)
ss_graphs(eco, model.her, model.cfg.cfg_graph)