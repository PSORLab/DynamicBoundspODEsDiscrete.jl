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
      Reexport, LinearAlgebra, IntervalArithmetic, StaticArrays,
      TaylorSeries, Requires, ElasticArrays, DataStructures, Polynomials

using ForwardDiff: Chunk, Dual, Partials, construct_seeds, single_seed,
      JacobianConfig, vector_mode_dual_eval, value, vector_mode_jacobian!,
      jacobian!
#using DiffEqSensitivity: extract_local_sensitivities, ODEForwardSensitivityProblem
#using OrdinaryDiffEq: ImplicitEuler, Trapezoid, ABDF2
#using DiffEqBase: remake, AbstractODEProblem, AbstractContinuousCallback, solve
using DiffResults: JacobianResult, MutableDiffResult

import DynamicBoundsBase: relax!, set!, setall!, get, getall!, getall, relax!, supports
import Base: setindex!, getindex, copyto!, literal_pow

import Base.MathConstants.golden

#import DynamicBoundsBase: relax!, integrate!
#export Wilhelm2019, set, setall!, get, getall!, relax!, integrate!

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
μ!(out,xⱼ,x̂ⱼ,η)

Used to compute the arguments of Jacobians (`x̂ⱼ + η(xⱼ - x̂ⱼ)`) used by the parametric Mean Value
Theorem. The result is stored to `out`.
"""
function μ!(out::Vector{Interval{Float64}}, xⱼ::Vector{Interval{Float64}}, x̂ⱼ::Vector{Float64}, η::Interval{Float64})
    out .= xⱼ
    return nothing
end
function μ!(out::Vector{MC{N,T}}, xⱼ::Vector{MC{N,T}}, x̂ⱼ::Vector{Float64}, η::Interval{Float64}) where {N, T<:RelaxTag}
    @__dot__ out = x̂ⱼ + η*(xⱼ - x̂ⱼ)
    return nothing
end

"""
ρ!(out,p,p̂ⱼ,η)

Used to compute the arguments of Jacobians (`p̂ⱼ + η(p - p̂ⱼ)`) used by the parametric Mean Value Theorem.
The result is stored to `out`.
"""
function ρ!(out::Vector{Interval{Float64}}, p::Vector{Interval{Float64}}, p̂::Vector{Float64}, η::Interval{Float64})
    out .= p
    return nothing
end
function ρ!(out::Vector{MC{N,T}}, p::Vector{MC{N,T}}, p̂::Vector{Float64}, η::Interval{Float64}) where {N, T<:RelaxTag}
    @__dot__ out = p̂ + η*(p - p̂)
    return nothing
end

include("StaticTaylorSeries/StaticTaylorSeries.jl")
using .StaticTaylorSeries

include("DiscretizeRelax/utilities/mul_split.jl")
include("DiscretizeRelax/utilities/fast_set_index.jl")
include("DiscretizeRelax/utilities/qr_utilities.jl")
include("DiscretizeRelax/utilities/coeff_calcs.jl")

include("DiscretizeRelax/utilities/taylor_functor.jl")
include("DiscretizeRelax/utilities/jacobian_functor.jl")
include("DiscretizeRelax/utilities/single_step.jl")

include("DiscretizeRelax/method/higher_order_enclosure.jl")
include("DiscretizeRelax/method/lohners_qr.jl")
include("DiscretizeRelax/method/hermite_obreschkoff.jl")

include("DiscretizeRelax/utilities/discretize_relax.jl")
include("DiscretizeRelax/utilities/relax.jl")
include("DiscretizeRelax/utilities/access_functions.jl")

#include("local_integration_problem.jl")
#include("interval_contractor.jl")
#include("Wilhelm2019.jl")

end # module
