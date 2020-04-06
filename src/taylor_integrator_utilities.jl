"""
$(TYPEDSIGNATURES)

A variant of the jetcoeffs! function used in TaylorIntegration.jl
(https://github.com/PerezHz/TaylorIntegration.jl/blob/master/src/explicitode.jl)
which preallocates taux and updates taux coefficients to avoid further allocations.

The TaylorIntegration.jl package is licensed under the MIT "Expat" License:
Copyright (c) 2016-2020: Jorge A. Perez and Luis Benet. Permission is hereby
granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following
conditions: The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
"""
function jetcoeffs!(eqsdiff!, t::Taylor1{T}, x::AbstractVector{Taylor1{U}},
                    dx::AbstractVector{Taylor1{U}}, xaux::AbstractVector{Taylor1{U}},
                    taux::AbstractVector{Taylor1{T}}, order::Int,
                    params) where {T<:Real, U<:Number}

      ordnext = 0
      for ord in 0:(order - 1)
          ordnext = ord + 1
          for i in 1:ordnext
              taux[ordnext].coeffs[i] = t.coeffs[i]
          end

          # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
          for j in eachindex(x)
              for i in 1:ordnext
                  xaux[j].coeffs[i] = x[j].coeffs[i]
              end
          end

          # Equations of motion
          eqsdiff!(dx, xaux, params, taux)

          # Recursion relations
          for j in eachindex(x)
              x[j].coeffs[ordnext+1] = dx[j].coeffs[ordnext]/Float64(ordnext)
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
    "Temporary storage for computing Ỹⱼ₀ in existence and uniqueness"
    Vⱼ::Vector{S}
    "Temporary storage for computing Taylor coefficient f̃ₜ in existence and uniqueness"
    f̃ₜ::Vector{S}
    "Temporary storage for computing Taylor coefficient f̃ in existence and uniqueness"
    f̃::Matrix{S}
    "Temporary storage for computing Ỹⱼ₀ in existence and uniqueness"
    Ỹⱼ₀::Vector{S}
    "Temporary storage for computing Ỹⱼ in existence and uniqueness"
    Ỹⱼ::Vector{S}
    "Temporary storage for computing ∂f∂y in existence and uniqueness"
    ∂f∂y::Vector{Matrix{S}}
    "Temporary storage for computing βⱼ in existence and uniqueness"
    βⱼⱼ::Matrix{S}
    "Temporary storage for computing βⱼ in existence and uniqueness"
    βⱼᵥ::Vector{S}
    "Temporary storage for computing βⱼ in existence and uniqueness"
    βⱼₖ::Vector{S}
    "Temporary storage for computing Ỹⱼ₀ in existence and uniqueness"
    Uⱼ::Vector{S}
    "In-place storage for Taylor coefficient input"
    xtaylor::Vector{Taylor1{S}}
    "In-place storage for Taylor coefficient output"
    xout::Vector{Taylor1{S}}
    "In-place temproary storage for Taylor coefficient calculation"
    xaux::Vector{Taylor1{S}}
    "State variables x"
    x::Vector{S}
    "Decision variables p"
    p::Vector{S}
end

"""
$(TYPEDSIGNATURES)

Computes the Taylor coefficients at `y` and stores them inplace to `out`.
"""
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
$(TYPEDSIGNATURES)

A constructor for `TaylorFunctor` that preallocates storage for computing
interval extensions of Taylor coefficients.
"""
function TaylorFunctor!(g!, nx::Int, np::Int, s::Int, t::T, q::Q) where {T <: Number, Q <: Number}
    @assert eltype(t) == Q
    taux = Taylor1{Q}[]
    for i in 0:(s-1)
        push!(taux, Taylor1(zeros(Q, i+1)))
    end
    x0 = zeros(T, nx)
    t = Taylor1(Q, s)
    Vⱼ = zeros(T, nx)
    f̃ₜ = zeros(T, nx*(s+1))
    f̃ = zeros(T, nx, s+1)
    Ỹⱼ₀ = zeros(T, nx + np)
    Ỹⱼ = zeros(T, nx + np)
    ∂f∂y = fill(zeros(T,nx,nx), s+1)
    βⱼⱼ = zeros(T,nx,nx)
    βⱼᵥ = zeros(T, nx)
    βⱼₖ = zeros(T, nx)
    Uⱼ = zeros(T, nx)
    xtaylor = Taylor1.(x0, s)
    xout = Taylor1.(x0, s)
    xaux = Taylor1.(x0, s)
    x = zeros(T, nx)
    p = zeros(T, np)
    return TaylorFunctor!{typeof(g!), Q, T}(g!, taux, t, nx, np,
                                            s, Vⱼ, f̃ₜ, f̃, Ỹⱼ₀, Ỹⱼ,
                                            ∂f∂y, βⱼⱼ, βⱼᵥ, βⱼₖ, Uⱼ,
                                            xtaylor, xout, xaux, x, p)
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
    "In-place storage for Taylor coefficient input"
    xtaylor::Vector{Taylor1{D}}
    "In-place storage for Taylor coefficient output"
    xout::Vector{Taylor1{D}}
    "In-place temporary storage for Taylor coefficient calculation"
    xaux::Vector{Taylor1{D}}
    "In-place temporary storage for Taylor coefficient calculation"
    out::Vector{S}
    "Variables y = (x,p)"
    y::Vector{S}
    "State variables x"
    x::Vector{D}
    "Decision variables p"
    p::Vector{D}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    B::Matrix{T}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    Δⱼ₊₁::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    Yⱼ₊₁::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    yⱼ₊₁::Vector{T}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    Rⱼ₊₁::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    mRⱼ₊₁::Vector{T}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    vⱼ₊₁::Vector{T}
    "P - p"
    rP::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    M1::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    M2::Matrix{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    M2Y::Matrix{S}
    "Storage for sum of Jacobian w.r.t x"
    Jxsto::Matrix{S}
    "Storage for sum of Jacobian w.r.t p"
    Jpsto::Matrix{S}
    "Temporary for transpose of Jacobian w.r.t y"
    tjac::Matrix{S}
    "Jacobian Configuration for ForwardDiff"
    cfg
end

"""
$(TYPEDSIGNATURES)

A constructor for TaylorFunctor that preallocates storage for computing interval
extensions of Taylor coefficients. The type `T` should use type `Q` for internal
computations.
"""
function JacTaylorFunctor!(g!, nx::Int, np::Int, s::Int, t::T, q::Q) where {T <: Number, Q <: Number}
    @assert eltype(t) == Q
    taux = Taylor1{Q}[]
    for i in 0:(s-1)
        push!(taux, Taylor1(zeros(Q, i+1)))
    end
    x0 = zeros(T, nx)
    xd0 = zeros(Dual{Nothing, T, nx+np}, nx)
    t = Taylor1(Q, s)
    xtaylor = Taylor1.(xd0, s)
    xout = Taylor1.(xd0, s)
    xaux = Taylor1.(xd0, s)
    out = zeros(T, nx*(s+1))
    y = zeros(T, nx + np)
    x = zeros(Dual{Nothing, T, nx+np}, nx)
    p = zeros(Dual{Nothing, T, nx+np}, np)
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
    Jxsto = zeros(T, nx, nx)
    Jpsto = zeros(T, nx, np)
    tjac = zeros(T, np + nx, nx*(s+1))
    cfg = JacobianConfig(nothing, out, zeros(T, nx + np))
    return JacTaylorFunctor!{typeof(g!), Q, T,
                             Dual{Nothing, T, nx+np}}(g!, taux, t,
                             nx, np, s, xtaylor, xout, xaux, out, y, x, p, B,
                             Δⱼ₊₁, Yⱼ₊₁, yⱼ₊₁, Rⱼ₊₁, mRⱼ₊₁, vⱼ₊₁, rP, M1, M2, M2Y,
                             Jxsto, Jpsto, tjac, cfg)
end

"""
$(TYPEDSIGNATURES)

Defines the call to `JacTaylorFunctor!` that preallocates storage to `Taylor1`
objects as necessary.
"""
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
                                 g::JacTaylorFunctor!, yin::Vector{S}) where {S <: Real}

    # copyto! is used to avoid allocations
    copyto!(g.y, 1, yin, 1, g.nx + g.np)

    # other AD schemes may be usable as well but this is a length(g.out) >> nx + np
    # situtation typically
    jacobian!(result, g, g.out, g.y, g.cfg)

    # reset sum of Jacobian storage
    fill!(g.Jxsto, zero(S))
    fill!(g.Jpsto, zero(S))
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

"""
$(TYPEDEF)

Provides preallocated storage for the QR factorization, Q, and the inverse of Q.

$(TYPEDFIELDS)
"""
mutable struct QRDenseStorage
    "QR Factorization"
    factorization::LinearAlgebra.QR{Float64,Array{Float64,2}}
    "Orthogonal matrix Q"
    Q::Array{Float64,2}
    "Inverse of Q"
    inv::Array{Float64,2}
end

"""
$(TYPEDSIGNATURES)

A constructor for QRDenseStorage assumes `Q` is of size `nx`-by-`nx` and of
type `Float64`.
"""
function QRDenseStorage(nx::Int)
    A = Float64.(Matrix(I, nx, nx))
    factorization = LinearAlgebra.qrfactUnblocked!(A)
    Q = similar(A)
    inverse = similar(A)
    QRDenseStorage(factorization, Q, inverse)
end

"""
$(TYPEDSIGNATURES)

Copies information in `y` to `x` in place.
"""
function Base.copyto!(x::QRDenseStorage, y::QRDenseStorage)
    x.factorization = y.factorization
    x.Q .= y.Q
    x.inv .= y.inv
    nothing
end

"""
$(TYPEDSIGNATURES)

Computes the QR factorization of `A` of size `(nx,nx)` and then stores it to
fields in `qst`.
"""
function calculateQ!(qst::QRDenseStorage, A::Matrix{Float64}, nx::Int)
    qst.factorization = LinearAlgebra.qrfactUnblocked!(A)
    qst.Q .= qst.factorization.Q*Matrix(I,nx,nx)
    nothing
end

"""
$(TYPEDSIGNATURES)

Computes `inv(Q)`` via transpose! and stores this to `qst.inverse`.
"""
function calculateQinv!(qst::QRDenseStorage)
    transpose!(qst.inv, qst.Q)
    nothing
end
