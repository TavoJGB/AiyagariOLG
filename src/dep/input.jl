#===========================================================================
    HELPER FUNCTIONS
===========================================================================#

function get_object(pars, substr; typesubstr::String="Prefix")
    subset = subset_namedtuple(pars, substr; typesubstr)
    tipo = subset.tipo
    req_pars = get_required_parameters(tipo)
    return _get_object(tipo; getindex(subset, req_pars)...)
end

function get_required_parameters(tipo::DataType)
    tiposhort = split(string(tipo),".")[end]
    "get_$(tiposhort)_parameters()" |> Meta.parse |> eval
end
get_required_parameters(type::SolverType) = get_solver_parameters(type)

function _get_object(tipo::DataType; kwargs...)
    tiposhort = split(string(tipo),".")[end]
    eval(Meta.parse("_$(tiposhort)"))(; kwargs...)
end
_get_object(tipo::SolverType; kwargs...) = Solver(tipo; kwargs...)



#===========================================================================
    READ PARAMETERS OR RESULTS
===========================================================================#

function import_csv(filepath; comment='#', delim='=')
    raw = []
    open(filepath, "r") do io
        while !eof(io)
            line = readline(io)
            if line[1] == comment
                continue
            else
                push!(raw, split(line,delim))
            end
        end
    end
    return NamedTuple(Symbol.(getindex.(raw, 1)) .=> eval.(Meta.parse.(getindex.(raw, 2))))
end



#===========================================================================
    BUILD MAIN STRUCTURES FROM PARAMETERS
===========================================================================#

function build_model(
    filepath = BASE_FOLDER * "/Simulations/parameters/default_parameters.csv";
    save_pars::Bool=true,   # by default, save parameters in file
    outputpath = BASE_FOLDER * "/Simulations/parameters/latest_simulation.csv",
    kwargs...
)
    # println(pwd())
    # println(filepath)
    # Read parameters
    pars_file = import_csv(filepath)
    pars_code = NamedTuple(kwargs)
    pars = merge(pars_file, pars_code)  # merge parameters, prioritising those introduced in the command line
    # Write parameters in file
    save_pars && export_csv(outputpath, pars; delim='=')
    # Grids and processes
    process_z = get_object(pars, "_z"; typesubstr="Suffix")
    grid_a = get_object(pars, "_a"; typesubstr="Suffix")
    # Configuration of solvers
    cfg_r = get_object(pars, "cfg_r_")
    cfg_hh = get_object(pars, "cfg_hh_")
    cfg_distr = get_object(pars, "cfg_distr_")
    cfg_graph = _GraphConfig(; subset_namedtuple(pars, "cfg_graph_")...)
    # Build structures
    hh = Households(; process_z, grid_a, getindex(pars, get_preference_parameters())...)
    fm = Firms(; getindex(pars, get_firm_parameters())...)
    cfg = Configuration(cfg_r, cfg_hh, cfg_distr, cfg_graph)
    # Return structures
    return (; hh, fm, cfg)
end