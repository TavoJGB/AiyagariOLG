#===========================================================================
    WRITE IN CSV FILE
===========================================================================#

function export_csv(filepath, pars::NamedTuple; delim='=')
    # Need double quotes for strings
    ind_str = (typeof.(values(pars)) .<: String) |> collect
    newvals = values(pars) |> collect
    @. newvals[ind_str] = '"' * newvals[ind_str] * '"'
    # Print parameters in CSV file
    csvfile = open(filepath,"w")
    for par in pairs(NamedTuple(keys(pars) .=> newvals))
        write(csvfile, join(par,delim), "\n")
    end
    close(csvfile)
end



#===========================================================================
    PREPARE TO BE EXPORTED
===========================================================================#

# Main function
function exportable(vec::Vector)
    return merge(exportable.(vec)...)
end

# Exporting StatDistr
function exportable(sd::StatDistr{<:StatisticType, <:EconomicVariable})
    keyvar = string(get_symbol(sd))
    vec_keys = keyvar * "_" .* sd.labels
    vec_vals = sd.values
    return NamedTuple(Symbol.(vec_keys) .=> vec_vals)
end

# Exporting NamedTuple of stats
function exportable(nt::NamedTuple{<:Any, <:Tuple{Vararg{Stat}}})
    vec_keys = keys(nt)
    vec_vals = [x.value for x in nt]
    return NamedTuple(vec_keys .=> vec_vals)
end



#===========================================================================
    COMPARE RESULTS
===========================================================================#

function compare_results(vec_results::Vector{<:NamedTuple})
    return maximum(abs.(diff(collect.(values.(vec_results)))[1]))
end
function compare_results(vec_filepaths::Vector{<:String})
    vec_results = import_csv.(vec_filepaths)
    return compare_results(vec_results)
end
function compare_results(eco::Economía;
                         compfilepath::String=BASE_FOLDER * "/Simulations/results/test_simulation.csv",
                         kwargs...)
    vec_results = [ exportable([ss_summarise(eco), ss_distributional_analysis(eco; kwargs...)]),
                    import_csv(compfilepath) ]
    return compare_results(vec_results)
end