using AiyagariOLG

# Settings
model = build_model(; N_z=5, N_a=150);

# Compute steady state
eco = steady(model...; r_0=0.04);

# Display steady state
ss_analysis(eco, model.her; top=0.1)
ss_graphs(eco, model.her, model.cfg.cfg_graph)