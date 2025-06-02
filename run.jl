using AiyagariOLG

# Settings
model = build_model(; β=0.94, N_z=5, N_a=500);

# Compute steady state
@btime eco = steady(model...; r_0=0.04);
annualise!(eco)

# Display steady state
ss_analysis(eco; top=0.1)
ss_graphs(eco, model.cfg.cfg_graph)