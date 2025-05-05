using Aiyagari

# Settings
pars = model_parameters()               # model parameters (preferences, capital share, etc.) 
cfg = model_config(; N_z=5, N_a=100)    # solver configuration (grids, tolerances, etc.)

# Structures
her=Herramientas(; cfg.pars_grids..., pars.pars_z...)       # tools
hlds=Households(pars.pars_h, her)                           # households
prod=Firms(pars.pars_f...)                                  # producers

# Compute steady state
eco = steady(hlds, prod, her, cfg; r_0=0.04);