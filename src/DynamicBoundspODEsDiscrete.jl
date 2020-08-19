module DynamicBoundspODEsDiscrete

using McCormick, DocStringExtensions, DynamicBoundsBase,
      Reexport, LinearAlgebra, IntervalArithmetic, StaticArrays, TaylorSeries,
      ElasticArrays, DataStructures, Polynomials
#@reexport using DynamicBoundsBase

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

export DiscretizeRelax, AdamsMoulton, BDF, LohnerContractor, HermiteObreschkoff, PLMS

const DBB = DynamicBoundsBase

mutable struct PrintCount
      n::Int
end
PrintCount() = PrintCount(0)
function (x::PrintCount)(s::String)
      x.n += 1
      println("Sig #$(x.n): "*s)
      nothing
end

abstract type AbstractStateContractor end
abstract type AbstractStateContractorName end

function μ!(out::Vector{Interval{Float64}}, xⱼ::Vector{Interval{Float64}}, x̂ⱼ::Vector{Float64}, η::Interval{Float64})
    out .= xⱼ
    return nothing
end
function μ!(out::Vector{MC{N,T}}, xⱼ::Vector{MC{N,T}}, x̂ⱼ::Vector{Float64}, η::Interval{Float64}) where {N, T<:RelaxTag}
    @__dot__ out = x̂ⱼ + η*(xⱼ - x̂ⱼ)
    return nothing
end

function ρ!(out::Vector{Interval{Float64}}, p::Vector{Interval{Float64}}, p̂::Vector{Float64}, η::Interval{Float64})
    out .= p
    return nothing
end
function ρ!(out::Vector{MC{N,T}}, p::Vector{MC{N,T}}, p̂::Vector{Float64}, η::Interval{Float64}) where {N, T<:RelaxTag}
    @__dot__ out = p̂ + η*(p - p̂)
    return nothing
end


include("DiscretizeRelax/utilities/fast_set_index.jl")
include("DiscretizeRelax/utilities/qr_utilities.jl")
include("DiscretizeRelax/utilities/coeff_calcs.jl")

include("DiscretizeRelax/utilities/taylor_functor.jl")
include("DiscretizeRelax/utilities/jacobian_functor.jl")
include("DiscretizeRelax/utilities/single_step.jl")

include("DiscretizeRelax/method/higher_order_enclosure.jl")
include("DiscretizeRelax/method/lohners_qr.jl")
include("DiscretizeRelax/method/hermite_obreschkoff.jl")
#include("DiscretizeRelax/method/pilms.jl")

include("DiscretizeRelax/utilities/discretize_relax.jl")
include("DiscretizeRelax/utilities/relax.jl")
include("DiscretizeRelax/utilities/access_functions.jl")

#include("local_integration_problem.jl")
#include("interval_contractor.jl")
#include("Wilhelm2019.jl")

end # module
