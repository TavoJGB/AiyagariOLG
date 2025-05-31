using AiyagariOLG

# Settings
model = build_model(; N_z=5, N_a=250);

# Compute steady state
eco = steady(model...; r_0=0.04);
annualise!(eco)

# Display steady state
ss_analysis(eco; top=0.1)
ss_graphs(eco, model.cfg.cfg_graph)