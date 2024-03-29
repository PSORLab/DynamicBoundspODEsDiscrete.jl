# Copyright(c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# Dynamic Bounds - pODEs Discrete
# A package for discretize and relax methods for bounding pODEs.
# See https://github.com/PSORLab/DynamicBoundspODEsDiscrete.jl
#############################################################################
# src/DynamicBoundspODEsDiscrete.jl
# Defines main module.
#############################################################################

module DynamicBoundspODEsDiscrete

using McCormick, DocStringExtensions, DynamicBoundsBase,
      Reexport, LinearAlgebra, StaticArrays, ElasticArrays, Polynomials,
      UnPack

using ForwardDiff: Chunk, Dual, Partials, construct_seeds, single_seed,
      JacobianConfig, vector_mode_dual_eval!, value, vector_mode_jacobian!,
      jacobian!

using DiffEqSensitivity: extract_local_sensitivities, ODEForwardSensitivityProblem
using DiffEqBase: remake, AbstractODEProblem, AbstractContinuousCallback, solve
using Sundials
using OrdinaryDiffEq: ABDF2, Trapezoid, ImplicitEuler

using DiffResults: JacobianResult, MutableDiffResult

import DynamicBoundsBase: relax!, set!, setall!, get, getall!, getall, relax!,
                          integrate!, supports
import Base: setindex!, getindex, copyto!, literal_pow, copy

import Base.MathConstants.golden

export DiscretizeRelax, AdamsMoulton, BDF, LohnerContractor, HermiteObreschkoff

export StepParams, StepResult, ExistStorage, ContractorStorage, reinitialize!,
       existence_uniqueness!, improvement_condition, single_step!, set_xX!,
       state_contractor_steps, state_contractor_γ, state_contractor_k, excess_error,
       set_Δ!, compute_X0!, set_P!, contains, calc_alpha, mul_split!, μ!, ρ!

const DBB = DynamicBoundsBase

abstract type AbstractStateContractor end

"""
AbstractStateContractorName

The subtypes of `AbstractStateContractorName` are used
to specify the manner of contractor method to be used
by `DiscretizeRelax` in the discretize and relax scheme.
"""
abstract type AbstractStateContractorName end

"""
state_contractor_k(d::AbstractStateContractorName)

Retrieves the order of the existence test to
be used with
"""
function state_contractor_k(d::AbstractStateContractorName)
    error("No method with AbstractStateContractorName $d defined.")
end

"""
state_contractor_γ(d::AbstractStateContractorName)
"""
function state_contractor_γ(d::AbstractStateContractorName)
    error("No method with AbstractStateContractorName $d defined.")
end

"""
state_contractor_steps(d::AbstractStateContractorName)
"""
function state_contractor_steps(d::AbstractStateContractorName)
    error("No method with AbstractStateContractorName $d defined.")
end

"""
state_contractor_integrator(d::AbstractStateContractorName)
"""
function state_contractor_integrator(d::AbstractStateContractorName)
    error("No method with AbstractStateContractorName $d defined.")
end

"""
μ!(xⱼ,x̂ⱼ,η)

Used to compute the arguments of Jacobians (`x̂ⱼ + η(xⱼ - x̂ⱼ)`) used by the parametric Mean Value
Theorem. The result is stored to `out`.
"""
function μ!(z, xⱼ::Vector{Interval{Float64}}, x̂ⱼ::Vector{Float64}, η::Interval{Float64})
    @. z = xⱼ
end
function μ!(z, xⱼ::Vector{MC{N,T}}, x̂ⱼ::Vector{Float64}, η::Interval{Float64}) where {N, T<:RelaxTag}
    @. z = x̂ⱼ + η*(xⱼ - x̂ⱼ)
end

"""
ρ!(out,p,p̂ⱼ,η)

Used to compute the arguments of Jacobians (`p̂ⱼ + η(p - p̂ⱼ)`) used by the parametric Mean Value Theorem.
The result is stored to `out`.
"""
function ρ!(z, p::Vector{Interval{Float64}}, p̂::Vector{Float64}, η::Interval{Float64})
    @. z = p
end
function ρ!(z, p::Vector{MC{N,T}}, p̂::Vector{Float64}, η::Interval{Float64}) where {N, T<:RelaxTag}
    @. z = p̂ + η*(p - p̂)
end
include("StaticTaylorSeries/StaticTaylorSeries.jl")
using .StaticTaylorSeries

include("DiscretizeRelax/utilities/fixed_buffer.jl")
include("DiscretizeRelax/utilities/mul_split.jl")
include("DiscretizeRelax/utilities/fast_set_index.jl")
include("DiscretizeRelax/utilities/qr_utilities.jl")
include("DiscretizeRelax/utilities/coeff_calcs.jl")

include("DiscretizeRelax/utilities/taylor_functor.jl")
include("DiscretizeRelax/utilities/jacobian_functor.jl")
include("DiscretizeRelax/utilities/single_step.jl")

print_iteration(x) = x > 98

function contract_constant_state!(x::Vector{Interval{Float64}}, t::ConstantStateBounds)
    for i in 1:length(t.xL)
        xL = x[i].lo
        xU = x[i].hi
        xLc = t.xL[i]
        xUc = t.xU[i]
        if xL < xLc
            x[i] = Interval(xLc, xU)
        elseif xU > xUc
            x[i] = Interval(xL, xUc)
        elseif (xL < xLc) && (xU > xUc)
            x[i] = Interval(xLc, xUc)
        end
    end
    return
end

function contract_constant_state!(x::Vector{MC{N,T}}, t::ConstantStateBounds) where {N,T}
    for i in 1:length(t.xL)
        xmc = x[i]
        xL = xmc.Intv.lo
        xU = xmc.Intv.hi
        xLc = t.xL[i]
        xUc = t.xU[i]
        if xL < xLc
            x[i] = x[i] ∩ Interval(xLc, Inf)
        elseif xU > xUc
            x[i] = x[i] ∩ Interval(-Inf, xUc)
        elseif (xL < xLc) && (xU > xUc)
            x[i] = MC{N,T}(Interval(xLc, xUc))
        end
    end
    return
end

function subgradient_expansion_interval_contract!(out::Vector{MC{N,T}}, p, pL, pU) where {N,T}
    for i = 1:length(out)
        x = out[i]
        l = Interval(x.cv)
        u = Interval(x.cc)
        for j = 1:length(p)
            P = Interval(pL[j], pU[j])
            l += x.cv_grad[j]*(P - p[j])
            u += x.cc_grad[j]*(P - p[j])
        end
        lower_x = max(x.Intv.lo, l.lo) # l.lo
        upper_x = min(x.Intv.hi, u.hi) # u.hi
        out[i] = MC{N,T}(x.cv, x.cc, Interval{Float64}(lower_x, upper_x), x.cv_grad, x.cc_grad, false)
    end
    nothing
end
subgradient_expansion_interval_contract!(out, p, pL, pU) = nothing


include("DiscretizeRelax/method/higher_order_enclosure.jl")
include("DiscretizeRelax/method/lohners_qr.jl")
include("DiscretizeRelax/method/hermite_obreschkoff.jl")
include("DiscretizeRelax/method/wilhelm_2019.jl")
include("DiscretizeRelax/method/pilms.jl")

include("DiscretizeRelax/utilities/discretize_relax.jl")
include("DiscretizeRelax/utilities/relax.jl")
include("DiscretizeRelax/utilities/access_functions.jl")

end # module
