using AiyagariOLG

# Settings
model = build_model(; β=0.97, N_z=5, N_a=500, max_a=40, fiscal=true);    # with unemployment
# model = build_model(; N_z=5, N_a=500);    # without unemployment

# Compute steady state
eco = steady(model...; r_0=0.04);
annualise!(eco)

# Display steady state
ss_analysis(eco; top=0.1)
ss_graphs(eco.hh, eco.pr, model.cfg.cfg_graph)