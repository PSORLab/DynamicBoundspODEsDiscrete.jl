module DynamicBoundspODEsPILMS

using McCormick, DocStringExtensions, DynamicBoundsBase,
      Reexport, LinearAlgebra, IntervalArithmetic, StaticArrays, TaylorSeries,
      ElasticArrays
#@reexport using DynamicBoundsBase

using ForwardDiff: Chunk, Dual, Partials, construct_seeds, single_seed,
      JacobianConfig, vector_mode_dual_eval, value, vector_mode_jacobian!,
      jacobian!
#using DiffEqSensitivity: extract_local_sensitivities, ODEForwardSensitivityProblem
#using OrdinaryDiffEq: ImplicitEuler, Trapezoid, ABDF2
#using DiffEqBase: remake, AbstractODEProblem, AbstractContinuousCallback, solve
using DiffResults: JacobianResult, MutableDiffResult

import Base: setindex!, getindex, copyto!

#import DynamicBoundsBase: relax!, integrate!
#export Wilhelm2019, set, setall!, get, getall!, relax!, integrate!

include("taylor_integrator_utilities.jl")
include("higher_order_enclosure.jl")
include("lohners_qr.jl")

#include("local_integration_problem.jl")
#include("interval_contractor.jl")
#include("Wilhelm2019.jl")

end # module
