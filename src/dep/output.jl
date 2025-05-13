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

function compare_results()
end