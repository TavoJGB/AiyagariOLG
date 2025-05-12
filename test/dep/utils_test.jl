function quantiles_test(eco::Economía; tol=1e-6)
    quantiles = ss_distributional_analysis(eco)
    @test maximum(abs.(sum.(quantiles).-1)) < tol
end