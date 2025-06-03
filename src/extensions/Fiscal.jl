module Fiscal

#===========================================================================
    IMPORTS
===========================================================================#

# Types
using ..AiyagariOLG: AbstractStateVariables, AbstractHouseholds, AbstractAggregates
using ..AiyagariOLG: AbstractEconomy, Generation, Firms, Prices, TimeStructure

# Methods
using ..AiyagariOLG: get_object, assemble

# Methods to extend
import ..AiyagariOLG: update_w!

# Other
using ..AiyagariOLG: @unpack, dot



#===========================================================================
    TAXES
===========================================================================#

abstract type TaxStruc end
Base.Broadcast.broadcastable(tax::TaxStruc) = Ref(tax)

# Linear tax
struct TaxLinear <: TaxStruc
    τ::Float64  # Tax rate
end
function get_tax(gross::Tr, tax::TaxLinear) where {Tr<:Real}
    return tax.τ * gross
end
function get_tax(gross::Vector{Tr}, tax::TaxLinear) where {Tr<:Real}
    return tax.τ * gross
end

function get_net(gross::Tr, tax::TaxStruc) where {Tr<:Real}
    return gross - get_tax(gross, tax)
end
export get_net

# Tax with brackets
struct TaxBrackets <: TaxStruc
    τ_mrg::Vector{Float64}          # Marginal tax rate on each bracket
    thresholds::Vector{Float64}     # Upper threshold of each bracket (note that it includes )
    cumtax::Vector{Float64}         # Cumulative tax in each bracket
end
function TaxBrackets(τ_mrg::Vector{Tr1}, thresholds::Vector{Tr2}) where {Tr1<:Real, Tr2<:Real}
    T_bracket = τ_mrg[1:(end-1)] .* diff([0; thresholds])
    return TaxBrackets(τ_mrg, [-99; thresholds], [0; cumsum(T_bracket)])
end
function get_tax(gross::Tr, tax::TaxBrackets) where {Tr<:Real}
    @unpack τ_mrg, thresholds, cumtax = tax;
    # Find bracket
    bracket = searchsortedlast.(Ref(thresholds), gross)
    # Compute tax
    return cumtax[bracket] + τ_mrg[bracket]*(gross - thresholds[bracket])
end
function get_tax(gross::Vector{Tr}, tax::TaxBrackets) where {Tr<:Real}
    @unpack τ_mrg, thresholds, cumtax = tax;
    # Find bracket
    bracket = searchsortedlast.(Ref(thresholds), gross)
    # Initialise variable
    return @. cumtax[bracket] + τ_mrg[bracket]*(gross - thresholds[bracket])
end

# HSV tax
struct TaxHSV <: TaxStruc
    λ::Float64      # Level
    γ::Float64      # Progressivity
end
function TaxHSV()
    return TaxHSV(1.0, 0.0)
end
function get_tax(gross::Real, tax::TaxHSV)
    @unpack λ, γ = tax;
    # Compute tax
    return max(0.0, gross - λ*gross^(1-γ))
end
function get_tax(gross::Vector{Tr}, tax::TaxHSV) where {Tr<:Real}
    @unpack λ, γ = tax;
    # Compute tax
    return max.(0.0, gross - λ*gross.^(1-γ))
end
function get_net(gross::Real, tax::TaxHSV)
    @unpack λ, γ = tax;
    return min(gross, λ*gross^(1-γ))
end
function marginal(gross::Real, tax::TaxHSV)
    @unpack λ, γ = tax;
    return max(0.0, 1-(1-γ)*λ*gross^(-γ))
end



#===========================================================================
    TRANSFERS
===========================================================================#

abstract type TransferStruc end
export TransferStruc

# Unemployment Insurance
struct UnemploymentInsurance <: TransferStruc
    ρ::Float64      # Replacement rate
end
export UnemploymentInsurance
_UnemploymentInsurance(; ρ::Real) = UnemploymentInsurance(ρ)
get_UnemploymentInsurance_parameters() = [:ρ]
export _UnemploymentInsurance, get_UnemploymentInsurance_parameters
function get_transfers(S::AbstractStateVariables, w::Real, ui::UnemploymentInsurance)
    @unpack emp, ζ, z = S;
    @unpack ρ = ui;
    return @. ρ*w*ζ*z*(!emp)
end
export get_transfers



#===========================================================================
    GOVERNMENT
===========================================================================#

struct Government
    # Policy Instruments
    tax::TaxStruc
    trf::TransferStruc
end
export Government

get_tax_revenue(L::Real, w::Real, tax_labinc::TaxLinear) = get_tax(w*L, tax_labinc)
get_tax_revenue(taxable::Vector{<:Real}, distr::Vector{<:Real}, tax::TaxStruc) = dot(get_tax.(taxable, tax), distr)
export get_tax_revenue

# INPUT: building a government
function build_government(pars, gens::Vector{<:Generation})
    # Get labour supply
    distr = assemble(gens, :distr)
    L = dot( assemble(gens, g -> g.S.ζ*g.S.z.*g.S.emp), distr )
    # Get transfer structure
    trf = get_object(pars, "trf_")
    # Get tax structure
    τ = implied_τ_labinc_linear(gens, trf, L)
    tax = TaxLinear(τ)
    # Return the structure
    return Government(tax, trf)
end
export build_government



#===========================================================================
    ECONOMY
===========================================================================#

mutable struct FiscalAggregates <: AbstractAggregates
    # Households
    A::Real         # Total assets
    A0::Real        # Initial assets
    C::Real         # Consumption
    L::Real         # Labour supply
    # Firms
    K::Real         # Capital stock
    Y::Real         # Output
    # Government
    TR::Real        # Total tax revenue
    PE::Real        # Public expenditure
end
export FiscalAggregates

struct FiscalEconomy <: AbstractEconomy
    # Agents
    hh::AbstractHouseholds
    fm::Firms
    gb::Government
    # Prices
    pr::Prices
    # Aggregates
    agg::FiscalAggregates
    # Time structure
    time_str::TimeStructure
    # Basic initialiser
end
export FiscalEconomy



#===========================================================================
    GENERAL EQUILIBRIUM
===========================================================================#

function implied_τ_labinc_linear(gens::Vector{<:Generation}, trf::UnemploymentInsurance, L::Real)
    distr = assemble(gens, :distr)
    L_unemp = dot( assemble(gens, g -> g.S.ζ*g.S.z .* (.!g.S.emp)), distr )
    return trf.ρ * L_unemp / L
end



#===========================================================================
    END OF MODULE
===========================================================================#

end