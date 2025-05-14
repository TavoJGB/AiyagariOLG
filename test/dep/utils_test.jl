function quantiles_test(eco::Economía; tol=1e-6)
    # Test that the sum of the quantiles is equal to 1
    quintiles = ss_distributional_analysis(eco)
    @test maximum(abs.(sum.(quintiles).-1)) < tol
    # Test that the top 20% coincides with the richest quintile
    quantiles = ss_distributional_analysis(eco, top=0.2)
    @test maximum(abs.(last.(diff.(getproperty.(quantiles, :values))))) < tol
end