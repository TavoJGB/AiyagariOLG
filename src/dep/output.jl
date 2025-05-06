#===========================================================================
    WRITE PARAMETERS IN FILE
===========================================================================#

function write_parameters(filepath, pars::NamedTuple; delim=',')
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