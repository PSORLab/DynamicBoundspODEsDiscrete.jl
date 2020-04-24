@generated function truncuated_STaylor1(x::STaylor1{N,T}, v::Int) where {N,T}
    ex_calc = quote end
    append!(ex_calc.args, Any[nothing for i in 1:N])
    syms = Symbol[Symbol("c$i") for i in 1:N]
    for i = 1:N
        sym = syms[i]
        ex_line = :($(sym) = $i <= v ? x[$(i-1)] : zero($T))
        ex_calc.args[i] = ex_line
    end
    exout = :(($(syms[1]),))
    for i = 2:N
        push!(exout.args, syms[i])
    end
    return quote
               Base.@_inline_meta
               $ex_calc
               return STaylor1{N,T}($exout)
            end
end

@generated function copy_recurse(dx::STaylor1{N,T}, x::STaylor1{N,T}, ord::Int, ford::Float64) where {N,T}
    ex_calc = quote end
    append!(ex_calc.args, Any[nothing for i in 1:(N+1)])
    syms = Symbol[Symbol("c$i") for i in 1:N]
    ex_line = :(nv = dx[ord-1]/ford) #/ord)
    ex_calc.args[1] = ex_line
    for i = 0:(N-1)
        sym = syms[i+1]
        ex_line = :($(sym) = $i < ord ? x[$i] : (($i == ord) ? nv : zero(T))) # nv))
        ex_calc.args[i+2] = ex_line
    end
    exout = :(($(syms[1]),))
    for i = 2:N
        push!(exout.args, syms[i])
    end
    return quote
               Base.@_inline_meta
               $ex_calc
               return STaylor1{N,T}($exout)
            end
end

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
function jetcoeffs!(eqsdiff!, t::T, x::Vector{STaylor1{N,U}}, xaux::Vector{STaylor1{N,U}},
                    dx::Vector{STaylor1{N,U}}, order::Int, params,
                    vnxt::Vector{Int}, fnxt::Vector{Float64}) where {N, T<:Number, U<:Number}

      ttaylor = STaylor1(t, Val{N-1}())
      for ord=1:order

          fill!(vnxt, ord)
          fill!(fnxt, Float64(ord))

          map!(truncuated_STaylor1, xaux, x, vnxt)
          eqsdiff!(dx, xaux, params, ttaylor)::Nothing
          x .= copy_recurse.(dx, x, vnxt, fnxt)
      end
      nothing
end
#=
function jetcoeffs!(eqsdiff!, t::T, x::Vector{STaylor1{N,U}}, xaux::Vector{STaylor1{N,U}},
                    dx::Vector{STaylor1{N,U}}, order::Int, params::Vector{U},
                    vnxt::Vector{Int}, fnxt::Vector{Float64}) where {N, T<:Number, U<:Number}

      ttaylor = STaylor1(t, Val{N-1}())
      for ord=1:order
          fill!(vnxt, ord)
          fill!(fnxt, Float64(ord))

          map!(truncuated_STaylor1, xaux, x, vnxt)
          eqsdiff!(dx, xaux, params, ttaylor)::Nothing
          x .= copy_recurse.(dx, x, vnxt, fnxt)
      end
      nothing
end
=#

"""
$(TYPEDEF)

A function g!(out, y) that perfoms a Taylor coefficient calculation. Provides
preallocated storage. Evaluating this function out is a vector of length nx*(s+1)
where 1:(s+1) are the Taylor coefficients of the first component, (s+2):nx*(s+1)
are the Taylor coefficients of the second component, and so on.

$(TYPEDFIELDS)
"""
mutable struct TaylorFunctor!{F <: Function, N, T <: Real, S <: Real}
    "Right-hand side function for pODE which operates in place as g!(dx,x,p,t)"
    g!::F
    "Dimensionality of x"
    nx::Int
    "Dimensionality of p"
    np::Int
    "Order of TaylorSeries"
    k::Int
    "Temporary storage for computing Ỹⱼ₀ in existence and uniqueness"
    Vⱼ::Vector{S}
    "Temporary storage for computing Taylor coefficient f̃ₜ in existence and uniqueness"
    f̃::Vector{Vector{S}}
    "Temporary storage for computing Ỹⱼ₀ in existence and uniqueness"
    X̃ⱼ₀::Vector{S}
    "Temporary storage for computing Ỹⱼ in existence and uniqueness"
    X̃ⱼ::Vector{S}
    "Temporary storage for computing ∂f∂x in existence and uniqueness"
    ∂f∂x::Vector{Matrix{S}}
    "Temporary storage for computing ∂f∂p in existence and uniqueness"
    ∂f∂p::Vector{Matrix{S}}
    "Temporary storage for computing βⱼ in existence and uniqueness"
    βⱼⱼ::Matrix{S}
    "Temporary storage for computing βⱼ in existence and uniqueness"
    βⱼᵥ::Vector{S}
    "Temporary storage for computing βⱼ in existence and uniqueness"
    βⱼₖ::Vector{S}
    "Temporary storage for computing Ỹⱼ₀ in existence and uniqueness"
    Uⱼ::Vector{S}
    "State variables x"
    x::Vector{S}
    "Decision variables p"
    p::Vector{S}
    "Store temporary STaylor1 vector for calculations"
    xtaylor::Vector{STaylor1{N,S}}
    "Store temporary STaylor1 vector for calculations"
    xaux::Vector{STaylor1{N,S}}
    "Store temporary STaylor1 vector for calculations"
    dx::Vector{STaylor1{N,S}}
    taux::Vector{STaylor1{N,T}}
    vnxt::Vector{Int}
    fnxt::Vector{Float64}
end

"""
$(TYPEDSIGNATURES)

Computes the Taylor coefficients at `y` and stores them inplace to `out`.
"""
function (d::TaylorFunctor!{F, K, T, S})(out::Vector{Vector{S}}, x::Vector{S},
          p::Vector{S}, t::T) where {F <: Function, K, T <: Real, S <: Real}

    val = Val(K-1)
    for i in eachindex(d.xtaylor)
        d.xtaylor[i] = STaylor1(x[i], val)
    end

    jetcoeffs!(d.g!, t, d.xtaylor, d.xaux, d.dx, K-1, p, d.vnxt, d.fnxt)::Nothing
    for i in eachindex(out)
        for j in eachindex(d.xtaylor)
            out[i][j] = d.xtaylor[j][i-1]
        end
    end
    nothing
end

"""
$(TYPEDSIGNATURES)

A constructor for `TaylorFunctor` that preallocates storage for computing
interval extensions of Taylor coefficients.
"""
function TaylorFunctor!(g!, nx::Int, np::Int, k::Val{K}, t::T, q::Q) where {K, T <: Number, Q <: Number}
    #println("typeof(t): $(typeof(t))")
    #println("typeof(Q): $(typeof(Q))")
    #println("typeof(eltype(t)): $(eltype(t))")
    #println("typeof(eltype(Q)): $(eltype(Q))")

    #@assert eltype(t) == Q
    x0 = zeros(T, nx)
    Vⱼ = zeros(T, nx)
    f̃ = Vector{T}[]
    for i in 1:(K+1)
        push!(f̃, zeros(T, nx))
    end
    temp = STaylor1(zeros(T,K+1))
    xtaylor = STaylor1{K+1,T}[]
    xaux = STaylor1{K+1,T}[]
    dx = STaylor1{K+1,T}[]
    taux = STaylor1{K+1,Q}[]
    for i in 1:nx
        push!(xtaylor, temp)
        push!(xaux, temp)
        push!(dx, temp)
        push!(taux, zero(STaylor1{K,Q}))
    end
    #push!(xtaylor, fill(temp, nx))
    X̃ⱼ₀ = zeros(T, nx)
    X̃ⱼ = zeros(T, nx)
    ∂f∂x = fill(zeros(T,nx,nx), K+1)
    ∂f∂p = fill(zeros(T,nx,np), K+1)
    βⱼⱼ = zeros(T,nx,nx)
    βⱼᵥ = zeros(T, nx)
    βⱼₖ = zeros(T, nx)
    Uⱼ = zeros(T, nx)
    x = zeros(T, nx)
    p = zeros(T, np)
    vnxt = zeros(Int, nx)
    fnxt = zeros(Float64, nx)
    return TaylorFunctor!{typeof(g!), K+1, Q, T}(g!, nx, np, K, Vⱼ, f̃, X̃ⱼ₀, X̃ⱼ,
                                               ∂f∂x, ∂f∂p, βⱼⱼ, βⱼᵥ, βⱼₖ, Uⱼ, x,
                                               p, xtaylor, xaux, dx, taux, vnxt,
                                               fnxt)
end

"""
$(TYPEDEF)

A callable structure used to evaluate the Jacobian of Taylor cofficients. This
also contains some addition fields to be used as inplace storage when computing
and preconditioning paralleliped based methods to representing enclosure of the
pODEs (Lohner's QR, Hermite-Obreschkoff, etc.)

$(TYPEDFIELDS)
"""
mutable struct JacTaylorFunctor!{F <: Function, N, T <: Real, S <: Real, NY}
    "Right-hand side function for pODE which operates in place as g!(dx,x,p,t)"
    g!::F
    "Dimensionality of x"
    nx::Int
    "Dimensionality of p"
    np::Int
    "Order of TaylorSeries"
    s::Int
    "In-place temporary storage for Taylor coefficient calculation"
    out::Vector{S}
    "Variables y = (x,p)"
    y::Vector{S}
    "State variables x"
    x::Vector{Dual{Nothing,S,NY}}
    "Decision variables p"
    p::Vector{Dual{Nothing,S,NY}}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    B::Matrix{T}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    Δⱼ₊₁::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    Xⱼ₊₁::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    xⱼ₊₁::Vector{T}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    Rⱼ₊₁::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    mRⱼ₊₁::Vector{T}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    vⱼ₊₁::Vector{T}
    "Temporary storage nx-by-1 in Lohner's QR & Hermite-Obreschkoff"
    M1::Vector{S}
    "Temporary storage nx-by-1 in Lohner's QR & Hermite-Obreschkoff"
    M1a::Vector{S}
    "Temporary storage nx-by-1 in Lohner's QR & Hermite-Obreschkoff"
    M1b::Vector{S}
    "Temporary storage nx-by-nx in Lohner's QR & Hermite-Obreschkoff"
    M2::Matrix{S}
    "Temporary storage nx-by-nx in Lohner's QR & Hermite-Obreschkoff"
    M2a::Matrix{S}
    "Temporary storage nx-by-np in Lohner's QR & Hermite-Obreschkoff"
    M3::Matrix{S}
    "Temporary storage np-by-1 in Lohner's QR & Hermite-Obreschkoff"
    M4::Vector{S}
    "Temporary storage in Lohner's QR & Hermite-Obreschkoff"
    M2Y::Matrix{S}
    "Storage for sum of Jacobian w.r.t x"
    Jxsto::Matrix{S}
    "Storage for sum of Jacobian w.r.t p"
    Jpsto::Matrix{S}
    "Temporary for transpose of Jacobian w.r.t y"
    tjac::Matrix{S}
    "Storage for vector of Jacobian w.r.t x"
    Jx::Vector{Matrix{S}}
    "Storage for vector of Jacobian w.r.t p"
    Jp::Vector{Matrix{S}}
    "Jacobian Result from DiffResults"
    result::MutableDiffResult{1, Vector{S}, Tuple{Matrix{S}}}
    "Jacobian Configuration for ForwardDiff"
    cfg::JacobianConfig{Nothing,S,NY,Tuple{Vector{Dual{Nothing,S,NY}},Vector{Dual{Nothing,S,NY}}}}
    "Store temporary STaylor1 vector for calculations"
    xtaylor::Vector{STaylor1{N,Dual{Nothing,S,NY}}}
    "Store temporary STaylor1 vector for calculations"
    xaux::Vector{STaylor1{N,Dual{Nothing,S,NY}}}
    "Store temporary STaylor1 vector for calculations"
    dx::Vector{STaylor1{N,Dual{Nothing,S,NY}}}
    taux::Vector{STaylor1{N,T}}
    t::Float64
    vnxt::Vector{Int}
    fnxt::Vector{Float64}
end

"""
$(TYPEDSIGNATURES)

A constructor for TaylorFunctor that preallocates storage for computing interval
extensions of Taylor coefficients. The type `T` should use type `Q` for internal
computations.
"""
function JacTaylorFunctor!(g!, nx::Int, np::Int, k::Val{K}, t::T, q::Q) where {K, T <: Number, Q <: Number}
    #@assert eltype(t) == Q
    x0 = zeros(T, nx)
    xd0 = zeros(Dual{Nothing, T, nx+np}, nx)
    out = zeros(T, nx*(K+1))
    y = zeros(T, nx + np)
    x = zeros(Dual{Nothing, T, nx+np}, nx)
    p = zeros(Dual{Nothing, T, nx+np}, np)
    B = zeros(Q, nx, nx)
    Δⱼ₊₁ = zeros(T, nx)
    Xⱼ₊₁ = zeros(T, nx)
    xⱼ₊₁ = zeros(Q, nx)
    Rⱼ₊₁ = zeros(T, nx)
    mRⱼ₊₁ = zeros(Q, nx)
    vⱼ₊₁ = zeros(Q, nx)
    M1 = zeros(T, nx)
    M1a = zeros(T, nx)
    M1b = zeros(T, nx)
    M2 = zeros(T, nx, nx)
    M2a = zeros(T, nx, nx)
    M3 = zeros(T, nx, np)
    M4 = zeros(T, np)
    M2Y = zeros(T, nx, nx)
    Jxsto = zeros(T, nx, nx)
    Jpsto = zeros(T, nx, np)
    tjac = zeros(T, np + nx, nx*(K+1))
    cfg = JacobianConfig(nothing, out, zeros(T, nx + np))
    result = JacobianResult(out, zeros(T, nx + np))
    Jx = Matrix{T}[]
    Jp = Matrix{T}[]

    temp = zero(Dual{Nothing, T, nx+np})
    taux = [STaylor1(zero(Q), Val(K))]
    xtaylor = STaylor1.(xd0, Val(K))
    dx = STaylor1.(xd0, Val(K))
    xaux = STaylor1.(xd0, Val(K))
    for i in 1:(K+1)
        push!(Jx, zeros(T,nx,nx))
        push!(Jp, zeros(T,nx,np))
    end
    t = 0.0
    vnxt = zeros(Int, nx)
    fnxt = zeros(Float64, nx)
    return JacTaylorFunctor!{typeof(g!), K+1, Q, T, nx+np}(g!, nx, np, K, out,
                             y, x, p, B, Δⱼ₊₁, Xⱼ₊₁, xⱼ₊₁, Rⱼ₊₁, mRⱼ₊₁, vⱼ₊₁,
                             M1, M1a, M1b, M2, M2a, M3, M4, M2Y, Jxsto, Jpsto,
                             tjac, Jx, Jp, result, cfg, xtaylor, xaux, dx,
                             taux, t, vnxt, fnxt)
end

"""
$(TYPEDSIGNATURES)

Defines the call to `JacTaylorFunctor!` that preallocates storage to `Taylor1`
objects as necessary.
"""
function (d::JacTaylorFunctor!{F,K,T,S,NY})(out::AbstractVector{Dual{Nothing,S,NY}},
                                            y::AbstractVector{Dual{Nothing,S,NY}}) where {F <: Function,
                                                                                          K, T <: Real, S, NY}


    copyto!(d.x, 1, y, 1, d.nx)
    copyto!(d.p, 1, y, d.nx+1, d.np)
    val = Val{K-1}()
    for i=1:d.nx
        d.xtaylor[i] = STaylor1(d.x[i], val)
    end
    jetcoeffs!(d.g!, d.t, d.xtaylor, d.xaux, d.dx, K-1, d.p, d.vnxt, d.fnxt)
    for q=1:K
        for i=1:d.nx
            indx = d.nx*(q-1) + i
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
function jacobian_taylor_coeffs!(g::JacTaylorFunctor!{F,K,T,S,NY}, X::Vector{S}, P, t::T) where {F,K,T,S,NY}

    # copyto! is used to avoid allocations
    copyto!(g.y, 1, X, 1, g.nx)
    copyto!(g.y, g.nx + 1, P, 1, g.np)
    g.t = t
    # other AD schemes may be usable as well but this is a length(g.out) >> nx + np
    # situtation typically
    jacobian!(g.result, g, g.out, g.y, g.cfg)

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
function set_JxJp!(g::JacTaylorFunctor!{F,K,T,S,NY}, X::Vector{S}, P, t) where {F,K,T,S,NY}

    jacobian_taylor_coeffs!(g, X, P, t)
    jac = g.result.derivs[1]
    for i in 1:(g.s+1)
        for q in 1:g.nx
            for z in 1:g.nx
                g.Jx[i][z, q] = jac[q + g.nx*(i-1), z]
            end
            for z in 1:g.np
                g.Jp[i][z, q] = jac[q + g.nx*(i-1), g.nx + z]
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

Computes `inv(Q)` via transpose! and stores this to `qst.inverse`.
"""
function calculateQinv!(qst::QRDenseStorage)
    transpose!(qst.inv, qst.Q)
    nothing
end

"""
An circular buffer of fixed capacity and length which allows
for access via getindex and copying of an element to the last then cycling
the last element to the first and shifting all other elements. See
[DataStructures.jl](https://github.com/JuliaCollections/DataStructures.jl).
"""
function DataStructures.CircularBuffer(a::T, length::Int) where T
    cb = CircularBuffer{T}(length)
    append!(cb, [zero.(a) for i=1:length])
    cb
end

function eval_cycle!(f!, cb::CircularBuffer, x, p, t)
    cb.first = (cb.first == 1 ? cb.length : cb.first - 1)
    f!(cb.buffer[cb.first], x, p, t)
    nothing
end

"""
$(TYPEDSIGNATURES)

Creates preallocated storage for an array of QR factorizations.
"""
function qr_stack(nx::Int, steps::Int)
    qrstack = CircularBuffer{QRDenseStorage}(steps)
    vector = fill(QRDenseStorage(nx), steps)
    append!(qrstack, vector)
    qrstack
end

"""
$(TYPEDSIGNATURES)

Sets the first QR storage to the identity matrix.
"""
function reinitialize!(x::CircularBuffer{QRDenseStorage})
    fill!(x[1].Q, 0.0)
    for i in 1:size(x[1].Q, 1)
        x[1].Q[i,i] = 1.0
    end
    nothing
end

mutable struct UniquenessResult{S <: Number}
    step::Float64
    confirmed::Bool
    X::Vector{S}
    fk::Vector{S}
end
function UniquenessResult(s::S, nx::Int, np::Int) where S
    X = zeros(S, nx)
    fk = zeros(S, nx)
    UniquenessResult{S}(0.0, false, X, fk)
end

mutable struct StepResult{S <: Number}
    status_flag::TerminationStatusCode
    "User-specified step size (if h > 0.0)"
    h::Float64
    hj::Float64
    predicted_hj::Float64
    errⱼ::Float64
    xⱼ::Vector{Float64}
    zⱼ::Vector{S}
    Xⱼ::Vector{S}
    Xapriori::Vector{S}
    unique_result::UniquenessResult{S}
    f::Matrix{S}
    ∂f∂x::Vector{Matrix{S}}
    ∂f∂p::Vector{Matrix{S}}
    jacobians_set::Bool
end
function StepResult(s::S, nx::Int, np::Int, k::Int, h::Float64) where S
    status_flag = RELAXATION_NOT_CALLED
    hj = 0.0
    predicted_hj = 0.0
    errⱼ = 0.0
    xⱼ = zeros(Float64, nx)
    zⱼ = zeros(S, nx)
    Xⱼ = zeros(S, nx)
    Xapriori = zeros(S, nx)
    unique_result = UniquenessResult(s, nx, np)
    f = zeros(S, nx, k+1)
    ∂f∂x = Matrix{S}[zeros(S,nx,nx) for i in 1:(k+1)]
    ∂f∂p = Matrix{S}[zeros(S,nx,np) for i in 1:(k+1)]
    jacobians_set = true
    StepResult{S}(status_flag, h, hj, predicted_hj, errⱼ, xⱼ, zⱼ, Xⱼ, Xapriori,
                  unique_result, f, ∂f∂x, ∂f∂p, jacobians_set)
end

abstract type AbstractStateContractor end
abstract type AbstractStateContractorName end
