function ss_test(eco::Economía; tol=1e-6)
    # PRELIMINARIES
    # Unpack
    @unpack hh, fm, pr, agg = eco;
    @unpack gens, process_z = hh;
    @unpack A, A0, K, L, C, Y = agg;
    # Size variables
    N_z = size(process_z)
    # Assemble relevant variables
    distr = assemble(gens, :distr)
    izz = assemble(gens, :states, :z)

    # Q-MATRIX
    # Total transition probabilities for each combination of states (each column) is 1
    @test maximum(abs.( assemble(gens, g -> sum(getproperty(g,:Q), dims=1) .- 1.0 ))) < tol   

    # STATIONARY DISTRIBUTION
    # Total population is normalised to 1
    @test abs(sum(distr) - 1) < tol
    # Distribution is stationary (the stationary distribution of the newby is not checked)
    @test maximum(abs.(vcat([g.distr - g_prev.Q*g_prev.distr for (g,g_prev) in zip(ZipForward(), gens)]...))) < tol
    # SS distribution by productivity level
    ss_z_dist = vcat([sum(distr[indZ]) for indZ in eachcol(izz .== (1:N_z)')]...)
    @test maximum(abs.(ss_z_dist - process_z.ss_dist)) < tol

    # HOUSEHOLD'S PROBLEM
    # Euler equation (only applies to unconstrained households that are not at the end of their lifespan)
    errs_eu = err_euler(eco)
    unconstr = .!get_borrowing_constrained(gens[1:(end-1)])
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