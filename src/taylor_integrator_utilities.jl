using LinearAlgebra, IntervalArithmetic, StaticArrays, TaylorSeries,
      TaylorIntegration, ForwardDiff, McCormick, BenchmarkTools, DocStringExtension

using DiffResults: JacobianResult, MutableDiffResult
using ForwardDiff: Partials, JacobianConfig, vector_mode_dual_eval, value, vector_mode_jacobian!

import Base.copyto!

"""
$(TYPEDSIGNATURES)

A variant of the jetcoeffs! function used in TaylorIntegration.jl
(https://github.com/PerezHz/TaylorIntegration.jl/blob/master/src/explicitode.jl)
which preallocates taux and updates taux coefficients to avoid further allocations.
"""
function jetcoeffs!(eqsdiff!, t::Taylor1{T}, x::AbstractVector{Taylor1{U}},
                    dx::AbstractVector{Taylor1{U}}, xaux::AbstractVector{Taylor1{U}},
                    taux::AbstractVector{Taylor1{T}}, order::Int,
                    params) where {T<:Real, U<:Number}

      ordnext = 0
      for ord in 0:(order - 1)
          ordnext = ord + 1
          for i in 1:ordnext
              @inbounds taux[ordnext].coeffs[i] = t.coeffs[i]
          end

          # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
          for j in eachindex(x)
              for i in 1:ordnext
                  @inbounds xaux[j].coeffs[i] = x[j].coeffs[i]
              end
          end

          # Equations of motion
          eqsdiff!(dx, xaux, params, taux)

          # Recursion relations
          for j in eachindex(x)
              @inbounds x[j].coeffs[ordnext+1] = dx[j].coeffs[ordnext]/Float64(ordnext)
          end
      end
  nothing
end

"""
$(TYPEDEF)

A function g!(out, y) that perfoms a Taylor coefficient calculation. Provides
preallocated storage. Evaluating this function out is a vector of length nx*(s+1)
where 1:(s+1) are the Taylor coefficients of the first component, (s+2):nx*(s+1)
are the Taylor coefficients of the second component, and so on.

$(TYPEDFIELDS)
"""
struct TaylorFunctor!{F <: Function, T <: Real, S <: Real} <: Function
    "Right-hand side function for pODE which operates in place as g!(dx,x,p,t)"
    g!::F
    "Storage for 1D Taylor series of t used in computing Taylor cofficients."
    taux::Vector{Taylor1{T}}
    "Storage for 1D Taylor series of t"
    t::Taylor1{T}
    "Dimensionality of x"
    nx::Int
    "Dimensionality of p"
    np::Int
    "Order of TaylorSeries"
    s::Int
    Vⱼ::Vector{S}
    f̃ₜ::Vector{S}
    f̃::Matrix{S}
    Ỹⱼ₀::Vector{S}
    Ỹⱼ::Vector{S}
    ∂f∂y::Vector{Matrix{S}}
    βⱼⱼ::Matrix{S}
    βⱼᵥ::Vector{S}
    βⱼₖ::Vector{S}
    Uⱼ::Vector{S}
    xtaylor::Vector{Taylor1{S}}
    xout::Vector{Taylor1{S}}
    xaux::Vector{Taylor1{S}}
    x::Vector{S}
    p::Vector{S}
end
function (d::TaylorFunctor!{F, T, S})(out, y) where {F <: Function, T <: Real, S}
    s = d.s
    x = d.x
    p = d.p
    nx = d.nx
    np = d.np
    copyto!(x, 1, y, 1, nx)
    copyto!(p, 1, y, nx+1, np)
    for i in 1:nx
        d.xtaylor[i].coeffs[1] = x[i]
        d.xout[i].coeffs[1] = x[i]
        d.xaux[i].coeffs[1] = x[i]
    end
    jetcoeffs!(d.g!, d.t, d.xtaylor, d.xout, d.xaux, d.taux, s, p)
    for i in 1:nx
        copyto!(out, (i - 1)*(s + 1) + 1, d.xtaylor[i].coeffs, 1, s + 1)
    end
    nothing
end

"""
A constructor for TaylorFunctor that preallocates storage for computing interval
extensions of Taylor coefficients.
"""
function TaylorFunctor!(g!, nx::Int, np::Int, s::Int)
    taux = Taylor1{Float64}[]
    for i in 0:(s-1)
        push!(taux, Taylor1(zeros(i+1)))
    end
    x0 = zeros(Interval{Float64}, nx)
    t = Taylor1(Float64, s)
    Vⱼ = zeros(Interval{Float64}, nx)
    f̃ₜ = zeros(Interval{Float64}, nx*(s+1))
    f̃ = zeros(Interval{Float64}, nx, s+1)
    Ỹⱼ₀ = zeros(Interval{Float64}, nx + np)
    Ỹⱼ = zeros(Interval{Float64}, nx + np)
    ∂f∂y = fill(zeros(Interval{Float64},nx,nx), s+1)
    βⱼⱼ = zeros(Interval{Float64},nx,nx)
    βⱼᵥ = zeros(Interval{Float64}, nx)
    βⱼₖ = zeros(Interval{Float64}, nx)
    Uⱼ = zeros(Interval{Float64}, nx)
    xtaylor = Taylor1.(x0, s)
    xout = Taylor1.(x0, s)
    xaux = Taylor1.(x0, s)
    x = zeros(Interval{Float64}, nx)
    p = zeros(Interval{Float64}, np)
    return TaylorFunctor!{typeof(g!), Float64, Interval{Float64}}(g!, taux, t, nx, np,
                                                                  s, Vⱼ, f̃ₜ, f̃, Ỹⱼ₀, Ỹⱼ,
                                                                  ∂f∂y, βⱼⱼ, βⱼᵥ, βⱼₖ, Uⱼ,
                                                                  xtaylor, xout, xaux,
                                                                  x,p)
end

"""
A constructor for TaylorFunctor that preallocates storage for computing float64
data type Taylor coefficients.
"""
function real_TaylorFunctor!(g!, nx::Int, np::Int, s::Int)
    taux = Taylor1{Float64}[]
    for i in 0:(s-1)
        push!(taux, Taylor1(zeros(i+1)))
    end
    x0 = zeros(Float64, nx)
    t = Taylor1(Float64, s)
    Vⱼ = zeros(Float64, nx)
    f̃ₜ = zeros(Float64, nx*(s+1))
    f̃ = zeros(Float64, nx, s+1)
    Ỹⱼ₀ = zeros(Float64, nx + np)
    Ỹⱼ = zeros(Float64, nx + np)
    ∂f∂y = fill(zeros(Float64,nx,nx), s+1)
    βⱼⱼ = zeros(Float64,nx,nx)
    βⱼᵥ = zeros(Float64, nx)
    βⱼₖ = zeros(Float64, nx)
    Uⱼ = zeros(Float64, nx)
    xtaylor = Taylor1.(x0, s)
    xout = Taylor1.(x0, s)
    xaux = Taylor1.(x0, s)
    x = zeros(Float64, nx)
    p = zeros(Float64, np)
    return TaylorFunctor!{typeof(g!), Float64, Float64}(g!, taux, t, nx, np,
                                                        s, Vⱼ, f̃ₜ, f̃, Ỹⱼ₀, Ỹⱼ,
                                                        ∂f∂y, βⱼⱼ, βⱼᵥ, βⱼₖ, Uⱼ,
                                                        xtaylor, xout, xaux,
                                                        x,p)
end

"""
$(TYPEDSIGNATURES)

A utility function for assigning a the vector of Taylor coefficients `a` to a
2D array `out` for an `nx` component system approximate to order `s`. The
`a[1:(s+1)]` contains the Taylor coefficients for x₁ and so on. Upon execution
`out[1:nx, 1]` contains the Taylor coefficients of the zeroth Taylor
coefficients of x and so on.
"""
function coeff_to_matrix!(out::Array{T,2}, a::Vector{T}, nx::Int, s::Int) where {T <: Real}
    @inbounds for i in 1:nx*(s+1)
        out[i] = a[i]
    end
    nothing
end

"""
$(TYPEDEF)

A callable structure used to evaluate the Jacobian of Taylor cofficients. This
also contains some addition fields to be used as inplace storage when computing
and preconditioning paralleliped based methods to representing enclosure of the
pODEs (Lohner's QR, Hermite-Obreschkoff, etc.)

$(TYPEDFIELDS)
"""
struct JacTaylorFunctor!{F <: Function, T <: Real, S <: Real, D} <: Function
    "Right-hand side function for pODE which operates in place as g!(dx,x,p,t)"
    g!::F
    "Storage for 1D Taylor series of t used in computing Taylor cofficients."
    taux::Vector{Taylor1{T}}
    "Storage for 1D Taylor series of t"
    t::Taylor1{T}
    "Dimensionality of x"
    nx::Int
    "Dimensionality of p"
    np::Int
    "Order of TaylorSeries"
    s::Int
    xtaylor::Vector{Taylor1{D}}
    xout::Vector{Taylor1{D}}
    xaux::Vector{Taylor1{D}}
    out::Vector{S}
    y::Vector{S}
    x::Vector{D}
    p::Vector{D}
    B::Matrix{T}
    Δⱼ₊₁::Vector{S}
    Yⱼ₊₁::Vector{S}
    yⱼ₊₁::Vector{T}
    Rⱼ₊₁::Vector{S}
    mRⱼ₊₁::Vector{T}
    vⱼ₊₁::Vector{T}
    rP::Vector{S}
    M1::Vector{S}
    M2::Matrix{S}
    M2Y::Matrix{S}
end

"""
$(TYPEDSIGNATURES)

A constructor for TaylorFunctor that preallocates storage for computing interval
extensions of Taylor coefficients. The type `T` should do computation is
"""
function JacTaylorFunctor!(g!, nx::Int, np::Int, s::Int, t::T, q::Q) where {T <: Number, Q <: Number}
    @assert eltype(T) == Q
    taux = Taylor1{Q}[]
    for i in 0:(s-1)
        push!(taux, Taylor1(zeros(Q, i+1)))
    end
    x0 = zeros(T, nx)
    xd0 = zeros(ForwardDiff.Dual{Nothing, T, nx+np}, nx)
    t = Taylor1(Q, s)
    xtaylor = Taylor1.(xd0, s)
    xout = Taylor1.(xd0, s)
    xaux = Taylor1.(xd0, s)
    out = zeros(T, nx*(s+1))
    y = zeros(T, nx + np)
    x = zeros(ForwardDiff.Dual{Nothing, T, nx+np}, nx)
    p = zeros(ForwardDiff.Dual{Nothing, T, nx+np}, np)
    B = zeros(Q, nx,nx)
    Δⱼ₊₁ = zeros(T, nx)
    Yⱼ₊₁ = zeros(T, nx)
    yⱼ₊₁ = zeros(Q, nx)
    Rⱼ₊₁ = zeros(T, nx)
    mRⱼ₊₁ = zeros(Q, nx)
    vⱼ₊₁ = zeros(Q, nx)
    rP = zeros(T, np)
    M1 = zeros(T, nx)
    M2 = zeros(T, nx, nx)
    M2Y = zeros(T, nx, nx)
    return JacTaylorFunctor!{typeof(g!), Q, T,
                             ForwardDiff.Dual{Nothing, T, nx+np}}(g!, taux, t,
                             nx, np, s, xtaylor, xout, xaux, out, y, x, p, B,
                             Δⱼ₊₁, Yⱼ₊₁, yⱼ₊₁, Rⱼ₊₁, mRⱼ₊₁, vⱼ₊₁, rP, M1, M2, M2Y)
end
function (d::JacTaylorFunctor!{F, T, S, D})(out, y) where {F <: Function, T <: Real, S, D}
    s = d.s
    nx = d.nx
    np = d.np
    copyto!(d.x, 1, y, 1, nx)
    copyto!(d.p, 1, y, nx+1, np)
    for i in 1:d.nx
        d.xtaylor[i].coeffs[1] = d.x[i]
    end
    jetcoeffs!(d.g!, d.t, d.xtaylor, d.xout, d.xaux, d.taux, s, d.p)
    for q in 1:(s+1)
        for i in 1:nx
            indx = nx*(q-1) + i
            out[indx] = d.xtaylor[i].coeffs[q]
        end
    end
    nothing
end

"""
$(TYPEDSIGNATURES)

Computes the Jacobian of the Taylor coefficients w.r.t. y = (x,p) storing the
output inplace to `result`. A JacobianConfig object without tag checking, cfg,
is required input and is initialized from `cfg = ForwardDiff.JacobianConfig(nothing, out, y)`.
The JacTaylorFunctor! used for the evaluation is `g` and inputs are `x` and `p`.
"""
function jacobian_taylor_coeffs!(result::MutableDiffResult{1,Vector{S},Tuple{Matrix{S}}},
                                 g::JacTaylorFunctor!, x::Vector{S}, p::Vector{S},
                                 cfg::JacobianConfig{T,V,N}) where {S <: Real, T,V,N}

    # copyto! is used to avoid allocations
    copyto!(g.y, 1, x, 1, g.nx)
    copyto!(g.y, 1+nx, p, 1, g.np)

    # other AD schemes may be usable as well but this is a length(g.out) >> nx + np
    # situtation typically
    ForwardDiff.jacobian!(result, g, g.out, g.y, cfg)
    nothing
end

"""
$(TYPEDSIGNATURES)

Extracts the Jacobian of the Taylor coefficients w.r.t. x, `Jx`, and the
Jacobian of the Taylor coefficients w.r.t. p, `Jp`, from `result`. The order of
the Taylor series is `s`, the dimensionality of x is `nx`, the dimensionality of
p is `np`, and `tjac` is preallocated storage for the transpose of the Jacobian
w.r.t. y = (x,p).
"""
function extract_JxJp!(Jx::Vector{Matrix{T}}, Jp::Vector{Matrix{T}},
                       result::MutableDiffResult{1,Vector{T},Tuple{Matrix{T}}},
                       tjac::Matrix{T}, nx::Int, np::Int, s::Int) where {T <: Real}
    jac = result.derivs[1]
    transpose!(tjac, jac)
    for i in 1:(s+1)
        for q in 1:nx
            for z in 1:nx
                Jx[i][q,z] = tjac[z, q + nx*(i-1)]
            end
            for z in 1:np
                Jp[i][q,z] = tjac[z + nx, q + nx*(i-1)]
            end
        end
    end
    nothing
end

function f!(dx,x,p,t)
    dx[1] = x[1]
    dx[2] = x[2]
    nothing
end
np = 2
nx = 2
k = 3
x = [1.0; 2.0]
p = [2.2; 2.2]
y = Interval{Float64}.([x; p])
g = JacTaylorFunctor!(f!, nx, np, k)
out = g.out
cfg = ForwardDiff.JacobianConfig(nothing, out, y)
result = JacobianResult(out, y)
xIntv = Interval{Float64}.(x)
pIntv = Interval{Float64}.(p)
tcoeffs = jacobian_taylor_coeffs!(result, g, xIntv, pIntv, cfg)
#@btime jacobian_taylor_coeffs!($result, $g, $xIntv, $pIntv, $cfg)
jac = result.derivs[1]
tjac = zeros(Interval{Float64}, 4, 8)
val = result.value

Jx = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
Jp = Matrix{Interval{Float64}}[zeros(Interval{Float64},2,2) for i in 1:4]
extract_JxJp!(Jx, Jp, result, tjac, nx, np, k)
#@btime extract_JxJp!($Jx, $Jp, $result, $tjac, $nx, $np, $k)

# extact is good... actual jacobians look odd...
