function ss_test(eco::Economía; tol=1e-6)
    # PRELIMINARIES
    # Aggregate variables
    agg = Aggregates(eco)
    # Unpack
    @unpack hh, fm, pr, distr, Q = eco
    @unpack S, G, states, process_z = hh
    @unpack A, A0, K, L, C, Y = agg
    
    # Q-MATRIX
    # Total transition probabilities for each combination of states (each column) is 1
    @test maximum(abs.(sum(Q, dims=1) .- 1)) < tol

    # STATIONARY DISTRIBUTION
    # Total population is normalised to 1
    @test abs(sum(eco.distr) - 1) < tol
    # Distribution is stationary
    @test maximum(abs.(Q * distr - distr)) < tol
    # SS distribution by productivity level
    ss_z_dist = vcat([sum(distr[indJ]) for indJ in eachcol(states.z .== (1:size(process_z))')]...)
    @test maximum(abs.(ss_z_dist - process_z.ss_dist)) < tol

    # HOUSEHOLD'S PROBLEM
    # Euler equation (only applies to unconstrained households)
    errs_eu = err_euler(eco)
    unconstr = .!get_borrowing_constrained(eco)
    @test maximum(abs.(errs_eu[unconstr])) < 1000*tol  # lower tolerance requires larger N_a 
    # sum(distr .* (abs.(errs_eu).>100tol) .* (unconstr))
    # Budget constraint
    @test maximum(abs.(err_budget(eco))) < tol

    # MARKETS
    # Capital market: households' side
    @test abs(dot(G.a′, eco.distr) - A) < tol
    # Capital market: firms' side
    @test abs(fm.ratio_KL(pr.r)*L - K) < tol
    # Capital market: clearing
    @test abs(A - K) < tol
    # Capital market: stationarity
    @test abs(A - A0) < tol
    # Goods market: clearing
    @test abs(A + C - (1-fm.δ)A0 - Y) < tol
end