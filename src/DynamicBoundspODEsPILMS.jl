module DynamicBoundspODEsPILMS

using McCormick, DocStringExtensions, DynamicBoundsBase,
      Reexport, LinearAlgebra, IntervalArithmetic, StaticArrays, TaylorSeries,
      ElasticArrays, DataStructures
#@reexport using DynamicBoundsBase

using ForwardDiff: Chunk, Dual, Partials, construct_seeds, single_seed,
      JacobianConfig, vector_mode_dual_eval, value, vector_mode_jacobian!,
      jacobian!
#using DiffEqSensitivity: extract_local_sensitivities, ODEForwardSensitivityProblem
#using OrdinaryDiffEq: ImplicitEuler, Trapezoid, ABDF2
#using DiffEqBase: remake, AbstractODEProblem, AbstractContinuousCallback, solve
using DiffResults: JacobianResult, MutableDiffResult

import DynamicBoundsBase: relax!, set, setall!, get, getall!, getall, relax!, supports
import Base: setindex!, getindex, copyto!

#import DynamicBoundsBase: relax!, integrate!
#export Wilhelm2019, set, setall!, get, getall!, relax!, integrate!

export DiscretizeRelax, AdamsMoulton, BDF

const DBB = DynamicBoundsBase

mutable struct PrintCount
      n::Int
end
PrintCount() = PrintCount(0)
function (x::PrintCount)(s::String)
      x.n += 1
     # println("Sig #$(x.n): "*s)
      nothing
end

include("DiscretizeRelax/taylor_integrator_utilities.jl")
include("DiscretizeRelax/higher_order_enclosure.jl")
include("DiscretizeRelax/lohners_qr.jl")
include("DiscretizeRelax/validated_pilms.jl")
include("DiscretizeRelax/hermite_obreschkoff.jl")
include("DiscretizeRelax/validated_integrator.jl")
include("DiscretizeRelax/access_functions.jl")

#include("local_integration_problem.jl")
#include("interval_contractor.jl")
#include("Wilhelm2019.jl")

end # module
