#===========================================================================
    HELPER FUNCTIONS
===========================================================================#

function get_object(pars, prefix)
    subset = subset_namedtuple(pars, prefix)
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
    READ PARAMETERS
===========================================================================#

function read_parameters(filepath; comment='#', delim=',')
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
function build_model(
    filepath = BASE_FOLDER * "/parameters/default_parameters.csv";
    kwargs... # keyword arguments
)
    # Read parameters
    pars_file = read_parameters(filepath)
    pars_code = NamedTuple(kwargs)
    pars = merge(pars_file, pars_code)  # merge parameters, prioritising those introduced in the command line
    # Grids and processes
    process_z = get_object(pars, "proc_z_")
    grid_a = get_object(pars, "grid_a_")
    # Configuration of solvers
    cfg_r = get_object(pars, "cfg_r_")
    cfg_hh = get_object(pars, "cfg_hh_")
    cfg_distr = get_object(pars, "cfg_distr_")
    # Build structures
    her = Herramientas(; process_z, grid_a)
    hlds = Households(her; getindex(pars, get_household_parameters())...)
    prod = Firms(; getindex(pars, get_firm_parameters())...)
    cfg = Configuration(cfg_r, cfg_hh, cfg_distr)
    # Return structures
    return (hlds, prod, her, cfg)
end