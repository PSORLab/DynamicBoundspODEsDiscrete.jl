"""
$(TYPEDEF)

A structure that stores the cofficient of the (P,Q)-Hermite-Obreschkoff method.
(Offset due to method being zero indexed and Julia begin one indexed).
$(TYPEDFIELDS)
"""
struct HermiteObreschkoff{P,Q,K} <: AbstractStateContractorName
    "Cpq[i=1:p] index starting at i = 1 rather than 0"
    cpq::Vector{Float64}
    "Cqp[i=1:q] index starting at i = 1 rather than 0"
    cqp::Vector{Float64}
    "gamma for method"
    γ::Float64
    "Explicit order Hermite-Obreschkoff"
    p::Int
    "Implicit order Hermite-Obreschkoff"
    q::Int
    "Total order Hermite-Obreschkoff"
    k::Int
end
function HermiteObreschkoff(p::Val{P}, q::Val{Q}) where {P,Q}
    temp_cpq = 1.0
    temp_cqp = 1.0
    cpq = zeros(P+1)
    cqp = zeros(Q+1)
    cpq[1] = temp_cpq
    cqp[1] = temp_cqp
    for i in 1:P
        temp_cpq *= (P - i + 1.0)/(P + Q - i + 1)
        cpq[i+1] = temp_cpq
    end
    γ = 1.0
    for i in 1:Q
        temp_cqp *= (Q - i + 1.0)/(Q + P - i + 1)
        cqp[i+1] = temp_cqp
        γ *= i/(P+i)
    end
    K = P + Q + 1
    HermiteObreschkoff{P,Q,K}(cpq, cqp, γ, P, Q, K)
end
HermiteObreschkoff(p::Int, q::Int) = HermiteObreschkoff(Val(p), Val(q))

"""
$(TYPEDEF)

A functor used to used to compute relaxations and bounds via Hermite-Obreschkoff methods.
"""
mutable struct HermiteObreschkoffFunctor{F <: Function, P, Q, K, Q1, T <: Real, S <: Real, NY} <: AbstractStateContractor
    hermite_obreschkoff::HermiteObreschkoff{P, Q, K}
    lon::LohnersFunctor{F, K, T, S, NY}
    implicit_r::TaylorFunctor!{F, Q1, T, T}
    implicit_J::JacTaylorFunctor!{F, Q1, T, S, NY}
    A::CircularBuffer{QRDenseStorage}
    Δⱼ::CircularBuffer{Vector{S}}
    η::Interval{T}
    μX::Vector{S}
    ρP::Vector{S}
    x̂0ⱼ₊₁::Vector{T}
    gⱼ₊₁::Vector{S}
    fqⱼ₊₁::Vector{S}
    fpⱼ₊₁::Vector{S}
    V1xS::Vector{S}
    V2xS::Vector{S}
    V3xS::Vector{S}
    V4xS::Vector{S}
    V5xS::Vector{S}
    V6xS::Vector{S}
    V7xS::Vector{S}
    V8xS::Vector{S}
    V9xS::Vector{S}
    M1xxT::Matrix{T}
    M1xxTa::Matrix{T}
    M1xxTb::Matrix{T}
    M1xxS::Matrix{S}
    M1xxSa::Matrix{S}
    M1xxSb::Matrix{S}
    M1xxSc::Matrix{S}
    M1xxSd::Matrix{S}
    M1xpS::Matrix{S}
end
function HermiteObreschkoffFunctor(f!::F, nx::Int, np::Int, p::Val{P}, q::Val{Q},
                                   k::Val{K}, s::S, t::T) where {F,P,Q,K,S,T}
    hermite_obreschkoff = HermiteObreschkoff(p,q)
    lon = LohnersFunctor(f!, nx, np, Val(K-1), s, t)
    implicit_r = TaylorFunctor!(f!, nx, np, Val(Q), zero(T), zero(T))
    implicit_J = JacTaylorFunctor!(f!, nx, np, Val(Q), zero(S), zero(T))
    A = CircularBuffer{QRDenseStorage}(2)
    Δⱼ = CircularBuffer{Vector{S}}(2)
    push!(A, QRDenseStorage(nx));  push!(A, QRDenseStorage(nx))
    push!(Δⱼ, zeros(S, nx)); push!(Δⱼ, zeros(S, nx))
    η = Interval{T}(0.0,1.0)
    μX = zeros(S, nx)
    ρP = zeros(S, np)
    x̂0ⱼ₊₁ = zeros(T, nx)
    gⱼ₊₁ = zeros(S, nx)
    fqⱼ₊₁ = zeros(S, nx)
    fpⱼ₊₁ = zeros(S, nx)
    V1xS = zeros(S, nx)
    V2xS = zeros(S, nx)
    V3xS = zeros(S, nx)
    V4xS = zeros(S, nx)
    V5xS = zeros(S, nx)
    V6xS = zeros(S, nx)
    V7xS = zeros(S, nx)
    V8xS = zeros(S, nx)
    V9xS = zeros(S, nx)
    M1xxT = zeros(T, nx, nx)
    M1xxTa = zeros(T, nx, nx)
    M1xxTb = zeros(T, nx, nx)
    M1xxS = zeros(S, nx, nx)
    M1xxSa = zeros(S, nx, nx)
    M1xxSb = zeros(S, nx, nx)
    M1xxSc = zeros(S, nx, nx)
    M1xxSd = zeros(S, nx, nx)
    M1xpS = zeros(S, nx, np)
    HermiteObreschkoffFunctor{F, P, Q, K, Q+1, T, S, nx+np}(hermite_obreschkoff, lon,
                                                       implicit_r, implicit_J, A, Δⱼ,
                                                       η, μX, ρP, x̂0ⱼ₊₁, gⱼ₊₁, fqⱼ₊₁,
                                                       fpⱼ₊₁, V1xS, V2xS, V3xS, V4xS, V5xS,
                                                       V6xS, V7xS, V8xS, V9xS, M1xxT, M1xxTa, M1xxTb,
                                                       M1xxS, M1xxSa, M1xxSb, M1xxSc, M1xxSd, M1xpS)
end

function state_contractor(m::HermiteObreschkoff{P,Q,K}, f, Jx!, Jp!, nx, np, style, s, h) where {P,Q,K}
    HermiteObreschkoffFunctor(f, nx, np, Val{P}(), Val{Q}(), Val{K}(), style, s)
end
state_contractor_k(m::HermiteObreschkoff{P,Q,K}) where {P,Q,K} = K
state_contractor_γ(m::HermiteObreschkoff) = m.γ
state_contractor_steps(m::HermiteObreschkoff) = 2

function mul_split!(Y::Vector{R}, A::Matrix{S}, B::Vector{T}, nx) where {R,S,T}
    if nx == 1
        @inbounds Y[1] = A[1,1]*B[1]
    else
        mul!(Y, A, B)
    end
    nothing
end

function mul_split!(Y::Matrix{R}, A::Matrix{S}, B::Matrix{T}, nx) where {R,S,T}
    if nx == 1
        @inbounds Y[1,1] = A[1,1]*B[1,1]
    else
        mul!(Y, A, B)
    end
    nothing
end

function copy_buffer!(y::CircularBuffer{T}, x::CircularBuffer{T}) where T
    y.capacity = x.capacity
    y.first = x.first
    y.length = x.length
    copyto!(y.buffer, x.buffer)
    nothing
end

# Hermite Obreschkoff Update #1
"""
$(TYPEDSIGNATURES)

Implements the a parametric version of Nedialkov's Hermite=Obreschkoff
method (based on Nedialko S. Nedialkov. Computing rigorous bounds on the solution of
an initial value problem for an ordinary differential equation. 1999. Universisty
of Toronto, PhD Dissertation, Algorithm 5.1, page 49) full details to be included
in a forthcoming paper.
"""
function (d::HermiteObreschkoffFunctor{F,Pp,Q,K,Q1,T,S,NY})(hbuffer::CircularBuffer{Float64},
                                                         tbuffer::CircularBuffer{Float64},
                                                         X̃ⱼ::Vector{S},
                                                         Xⱼ::Vector{S}, xval::Vector{T},
                                                         A::CircularBuffer{QRDenseStorage},
                                                         Δⱼ::CircularBuffer{Vector{S}}, P::Vector{S}, rP::Vector{S},
                                                         pval::Vector{T}, fk::Vector{S}) where {F, Pp, Q, K, Q1, T, S, NY}

    copy_buffer!(d.Δⱼ, Δⱼ)
    copy_buffer!(d.A, A)

    # Compute lohner function step
    implicitJf! = d.implicit_J
    explicitJf! = d.lon.jac_tf!
    explicitrf! = d.lon.real_tf!
    p = d.hermite_obreschkoff.p
    q = d.hermite_obreschkoff.q
    cqp = d.hermite_obreschkoff.cqp
    cpq = d.hermite_obreschkoff.cpq
    hⱼ = hbuffer[1]
    t = tbuffer[1]
    nx = d.lon.set_tf!.nx

    # perform a lohners method tightening
    d.lon(hbuffer, tbuffer, X̃ⱼ, Xⱼ, xval, A, Δⱼ, P, rP, pval, fk)

    fk .*= hⱼ^(p+q+1)

    Xⱼ₊₁ = explicitJf!.Xⱼ₊₁
    @__dot__ d.x̂0ⱼ₊₁ = mid(Xⱼ₊₁)

    # compute real value sum of taylor series (implicit)
    copyto!(d.implicit_r.X̃ⱼ₀, 1, xval, 1, nx)
    copyto!(d.implicit_r.X̃ⱼ, 1, xval, 1, nx)

    fill!(d.fqⱼ₊₁, zero(S))
    d.implicit_r(d.implicit_r.f̃, d.x̂0ⱼ₊₁, pval, t)
    for i = 2:q+1
        coeff = cqp[i]*(-hⱼ)^(i-1)
        @__dot__ d.fqⱼ₊₁ -= coeff*d.implicit_r.f̃[i]
    end

    fill!(d.fpⱼ₊₁, zero(S))
    for i = 2:p+1
        @__dot__ d.fpⱼ₊₁ += cpq[i]*explicitrf!.f̃[i]
    end
    @__dot__ d.gⱼ₊₁ = xval - d.x̂0ⱼ₊₁ + d.fpⱼ₊₁ + d.fqⱼ₊₁ + d.hermite_obreschkoff.γ*fk

    # compute sum of explicit Jacobian with ho weights
    fill!(explicitJf!.Jxsto, zero(S))
    fill!(explicitJf!.Jpsto, zero(S))
    for i = 1:p+1
        if i == 1
            for j = 1:nx
                explicitJf!.Jxsto[j,j] = one(S)
            end
        else
            coeff = cpq[i]*hⱼ^(i-1)
            @__dot__ explicitJf!.Jxsto += coeff*explicitJf!.Jx[i]
            @__dot__ explicitJf!.Jpsto += coeff*explicitJf!.Jp[i]
        end
    end

    # compute set-valued extension of Jacobian of Taylor series (implicit)
    μ!(d.μX, Xⱼ₊₁, xval, d.η)
    ρ!(d.ρP, P, pval, d.η)
    set_JxJp!(implicitJf!, d.μX, d.ρP, t)
    for i = 1:q+1
        if i == 1
            for j = 1:nx
                implicitJf!.Jxsto[j,j] = one(S)
            end
        else
            coeff = cqp[i]*(-hⱼ)^(i-1)
            @__dot__ implicitJf!.Jxsto += coeff*implicitJf!.Jx[i]
            @__dot__ implicitJf!.Jpsto += coeff*implicitJf!.Jp[i]
        end
    end

    @__dot__ d.M1xxT = mid(implicitJf!.Jxsto)
    invShat = inv(d.M1xxT)
    B0 = (invShat*explicitJf!.Jxsto)*d.A[1].Q
    mul_split!(d.M1xxS, invShat, implicitJf!.Jxsto, nx)
    @__dot__ d.M1xxS *= -one(T)
    for j = 1:nx
        d.M1xxS[j,j] += one(T)
    end
    @__dot__ d.V1xS = Xⱼ₊₁ - d.x̂0ⱼ₊₁
    @__dot__ explicitJf!.Jpsto -= implicitJf!.Jpsto
    mul_split!(d.M1xpS, invShat, explicitJf!.Jpsto, nx)
    mul_split!(d.V2xS, d.M1xpS, rP, nx)
    mul_split!(d.V3xS, d.M1xxTa, d.V1xS, nx)
    pre_intersect = B0*d.Δⱼ[1] + d.V3xS + invShat*d.gⱼ₊₁
    @__dot__ implicitJf!.Xⱼ₊₁ = (d.x̂0ⱼ₊₁ + pre_intersect  + d.V2xS) ∩ Xⱼ₊₁

    Aⱼ₊₁ = d.A[1]
    implicitJf!.B .= mid.(implicitJf!.Jxsto*d.A[1].Q)
    calculateQ!(Aⱼ₊₁, implicitJf!.B, nx)
    calculateQinv!(Aⱼ₊₁)

    @__dot__ implicitJf!.xⱼ₊₁ = mid(implicitJf!.Xⱼ₊₁)
    @__dot__ d.x̂0ⱼ₊₁ -= implicitJf!.xⱼ₊₁
    mul_split!(d.M1xxSa, Aⱼ₊₁.inv, d.M1xxS, nx)
    mul_split!(d.V4xS, d.M1xxTb, d.V1xS, nx)
    mul_split!(d.V7xS, Aⱼ₊₁.inv, d.x̂0ⱼ₊₁, nx)
    mul_split!(d.M1xxSb, Aⱼ₊₁.inv, B0, nx)
    mul_split!(d.V5xS, d.M1xxSb, d.Δⱼ[1], nx)
    mul_split!(d.M1xxSc, Aⱼ₊₁.inv, invShat, nx)
    mul_split!(d.V6xS, d.M1xxSc, d.gⱼ₊₁, nx)
    @__dot__ implicitJf!.Δⱼ₊₁ = d.V2xS + d.V4xS + d.V5xS + d.V6xS + d.V7xS

    pushfirst!(Δⱼ, implicitJf!.Δⱼ₊₁)
    RELAXATION_NOT_CALLED
end

function get_Δ(f::HermiteObreschkoffFunctor)
    f.implicit_J.Δⱼ₊₁
end
function set_x!(out::Vector{Float64}, f::HermiteObreschkoffFunctor)
    out .= f.implicit_J.xⱼ₊₁
    nothing
end
function set_X!(out::Vector{S}, f::HermiteObreschkoffFunctor) where S
    out .= f.implicit_J.Xⱼ₊₁
    nothing
end

has_jacobians(d::HermiteObreschkoffFunctor) = true

function extract_jacobians!(d::HermiteObreschkoffFunctor, ∂f∂x::Vector{Matrix{T}},
                            ∂f∂p::Vector{Matrix{T}}) where {T <: Real}
    for i = 1:d.lon.set_tf!.k+1
        ∂f∂x[i] .= d.lon.jac_tf!.Jx[i]
        ∂f∂p[i] .= d.lon.jac_tf!.Jp[i]
    end
    nothing
end

function get_jacobians!(d::HermiteObreschkoffFunctor, ∂f∂x::Vector{Matrix{T}},
                        ∂f∂p::Vector{Matrix{T}}, Xⱼ, P, t) where {T <: Real}
    set_JxJp!(d.lon.jac_tf!, Xⱼ, P, t[1])
    extract_jacobians!(d, ∂f∂x, ∂f∂p)
    nothing
end
